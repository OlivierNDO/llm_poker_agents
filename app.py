from flask import Flask, render_template, jsonify, request
import json
import os
from threading import Lock
import yaml

from src.game import Hand, Player, TableConfig, StatsTracker, TableManager, Action, ActionType
from src.agents import LLMAgent, RandomAgent
from src.agent_utils import OpenRouterCompletionEngine
from src.logging_config import logger

app = Flask(__name__)
game_lock = Lock()

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}

class PokerGameServer:
    def __init__(self):
        self.cfg = config
        game_cfg = self.cfg.get('game', {})
        self.stats_tracker = StatsTracker()
        self.config = TableConfig(
            small_blind=int(game_cfg.get('small_blind', 1)),
            big_blind=int(game_cfg.get('big_blind', 2)),
            max_hands=int(game_cfg.get('max_hands', 5))
        )
        self.players = []
        self.table_manager = None
        self.current_hand = None
        self.setup_players()
        
    def setup_players(self):
        """Initialize players from YAML config."""
        self.players = []
        players_cfg = self.cfg.get('players', [])
        for p in players_cfg:
            if not p.get('enabled', True):
                continue
            name = str(p.get('name', 'Player'))
            model_id = str(p.get('model', ''))
            chips = int(p.get('chips', 250))
            client = OpenRouterCompletionEngine(model=model_id)
            agent = LLMAgent(
                client=client,
                name=name,
                table_config=self.config,
                logger=logger
            )
            self.players.append(
                Player(
                    name=name,
                    chips=chips,
                    agent=agent
                )
            )
        
        # Create TableManager
        self.table_manager = TableManager(
            players=self.players,
            config=self.config,
            stats_tracker=self.stats_tracker,
            logger=logger
        )

    def _card_to_string(self, card):
        """Convert Card object to string representation like 'As', 'Kd', etc."""
        rank_map = {
            14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
             9: '9',  8: '8',  7: '7',  6: '6',  5: '5',
             4: '4',  3: '3',  2: '2'
        }
        suit_map = {
            0: 'c',  # clubs
            1: 'h',  # hearts  
            2: 'd',  # diamonds
            3: 's'   # spades
        }
        rank_char = rank_map.get(card.rank, str(card.rank))
        suit_char = suit_map.get(card.suit.value, 'c')
        return f"{rank_char}{suit_char}"

    def start_new_hand(self):
        """Start a new hand using TableManager"""
        # Check max hands limit
        if (getattr(self.config, 'max_hands', 0) and 
            self.table_manager.hand_no >= int(self.config.max_hands)):
            return {
                'error': 'Max hands reached',
                'complete': True,
                'handsPlayed': self.table_manager.hand_no,
                'maxHands': int(self.config.max_hands),
                'game_state': self.get_game_state() if self.current_hand else {}
            }
        
        logger.info(f"Starting new hand #{self.table_manager.hand_no + 1}")
        
        # Get active players
        active_players = [p for p in self.players if p.chips > 0]
        logger.info(f"Active players: {[p.name for p in active_players]}")
        
        if len(active_players) <= 1:
            logger.warning("Not enough players to start hand")
            return {"error": "Need at least 2 players"}
        
        # Reset players for new hand
        for player in active_players:
            player.reset_for_hand()
        logger.info("Players reset for new hand")
        
        # Create new hand
        self.current_hand = Hand(active_players, self.stats_tracker)
        logger.info("Hand object created")
        
        # Execute setup and deal hole cards FIRST (this resets GameState)
        logger.info("Executing hand setup...")
        self.current_hand.execute_next_step()  # setup - this creates new GameState with dealer_position = 0
        logger.info("Dealing hole cards...")
        self.current_hand.execute_next_step()  # deal hole cards
        
        # NOW set dealer position AFTER setup is complete
        dealer_player = self.players[self.table_manager.dealer_index]
        dealer_in_hand = next((i for i, p in enumerate(active_players) if p is dealer_player), 0)
        self.current_hand.game_state.dealer_position = dealer_in_hand
        logger.info(f"Dealer position in hand: {dealer_in_hand}")
        
        # Post blinds using TableManager
        logger.info("Posting blinds...")
        self.table_manager._post_blinds(self.current_hand)
        
        # Advance to preflop phase
        self.current_hand.phase = "preflop"
        logger.info("Hand setup complete, phase set to preflop")
        
        self.table_manager.hand_no += 1
        
        return self.get_game_state()

    def execute_next_action(self):
        """Execute the next step in the current hand"""
        if not self.current_hand:
            logger.error("Attempted to execute action with no active hand")
            return {"error": "No active hand"}
            
        if self.current_hand.is_complete:
            logger.info("Hand is already complete")
            return {"error": "Hand is complete"}
        
        logger.info(f"Executing next action in phase: {self.current_hand.phase}")
        
        try:
            result = self.current_hand.execute_next_step()
            logger.info(f"Step result: {result}")
            
            # If hand is complete, advance dealer button for next hand
            if self.current_hand.is_complete:
                logger.info("Hand completed, advancing dealer button")
                self.table_manager._advance_button()
            
            return {
                "step_result": result,
                "game_state": self.get_game_state(),
                "complete": self.current_hand.is_complete
            }
        except Exception as e:
            logger.error(f"Error executing next action: {e}", exc_info=True)
            return {"error": f"Failed to execute action: {str(e)}"}

    def auto_play_until_completion(self):
        """Run the game until completion using TableManager logic"""
        logger.info("Starting auto play until completion")
        
        if not self.current_hand:
            return {"error": "No active hand"}
        
        log_entries = []
        hands_played = 0
        max_hands_per_session = int(self.cfg.get('game', {}).get('max_hands_per_session', 500))
        
        try:
            while hands_played < max_hands_per_session:
                # Complete current hand if it exists and isn't finished
                if self.current_hand and not self.current_hand.is_complete:
                    logger.info(f"Completing current hand (hand #{self.table_manager.hand_no})")
                    
                    while not self.current_hand.is_complete:
                        step_result = self.current_hand.execute_next_step()
                        if step_result:
                            log_entries.append(f"Hand #{self.table_manager.hand_no}: {step_result}")
                    
                    # Advance dealer button after hand completion
                    if self.current_hand.is_complete:
                        self.table_manager._advance_button()
                        log_entries.append(f"Hand #{self.table_manager.hand_no} complete")
                
                # Check max hands limit BEFORE starting next hand
                if hasattr(self.config, 'max_hands') and self.config.max_hands > 0:
                    if self.table_manager.hand_no >= self.config.max_hands:
                        log_entries.append(f"Reached maximum hands limit ({self.config.max_hands})")
                        logger.info(f"Reached max hands limit: {self.config.max_hands}")
                        return {
                            "complete": True,
                            "game_state": self.get_game_state(),
                            "log_entries": log_entries,
                            "reason": "Max hands reached"
                        }
                
                # Check win condition - only one player with chips
                active_players = [p for p in self.players if p.chips > 0]
                if len(active_players) <= 1:
                    winner_name = active_players[0].name if active_players else "No winner"
                    log_entries.append(f"Game over! Winner: {winner_name}")
                    logger.info(f"Game completed - Winner: {winner_name}")
                    return {
                        "complete": True,
                        "game_state": self.get_game_state(),
                        "log_entries": log_entries,
                        "winner": winner_name
                    }
                
                # Start next hand
                logger.info(f"Starting hand #{self.table_manager.hand_no + 1}")
                next_hand_result = self.start_new_hand()
                
                if "error" in next_hand_result:
                    log_entries.append(f"Error starting new hand: {next_hand_result['error']}")
                    return {
                        "complete": True,
                        "game_state": self.get_game_state(),
                        "log_entries": log_entries,
                        "error": next_hand_result["error"]
                    }
                
                hands_played += 1
                log_entries.append(f"Started hand #{self.table_manager.hand_no}")
            
            # Safety limit reached
            log_entries.append(f"Reached safety limit of {max_hands_per_session} hands")
            return {
                "complete": True,
                "game_state": self.get_game_state(),
                "log_entries": log_entries,
                "reason": "Safety limit reached"
            }
            
        except Exception as e:
            logger.error(f"Error during auto play: {e}", exc_info=True)
            log_entries.append(f"Error during auto play: {str(e)}")
            return {
                "complete": True,
                "game_state": self.get_game_state(),
                "log_entries": log_entries,
                "error": str(e)
            }

    def get_game_state(self):
        """Convert current game state to JSON format for frontend"""
        if not self.current_hand:
            return {"error": "No active hand"}
        
        # Convert phase to readable name
        phase_names = {
            "setup": "Setup",
            "hole_cards": "Dealing", 
            "preflop": "Preflop",
            "flop": "Flop",
            "flop_betting": "Flop",
            "turn": "Turn", 
            "turn_betting": "Turn",
            "river": "River",
            "river_betting": "River",
            "showdown": "Showdown",
            "complete": "Complete"
        }
        
        # Get current player info
        current_player = None
        current_player_index = -1
        if hasattr(self.current_hand.game_state, 'current_player_index'):
            current_player_index = self.current_hand.game_state.current_player_index
            current_player = self.current_hand.game_state.get_current_player()
            logger.info(f"DEBUG: current_player = {current_player.name if current_player else None}")
            logger.info(f"DEBUG: current_player_index = {current_player_index}")
            logger.info(f"DEBUG: hand phase = {self.current_hand.phase}")
                    
        # Convert players to frontend format
        players_data = []
        hand_players = self.current_hand.players
        
        for i, player in enumerate(hand_players):
            # Get hole cards - convert Card objects to strings
            hole_cards = []
            if player.hole_cards:
                hole_cards = [f"{self._card_to_string(card)}" for card in player.hole_cards]
            
            # Get last action and reasoning from recent game history
            last_action = ""
            reasoning = ""
            recent_actions = self.current_hand.game_state.get_player_actions(player.name)
            if recent_actions:
                last_action_record = recent_actions[-1]
                logger.info(f"DEBUG: {player.name} last action record: {last_action_record}")
                
                # Extract the action string properly
                if hasattr(last_action_record.action, 'action_type'):
                    action_type = last_action_record.action.action_type
                    if action_type.value == 'fold':  # Use .value to get the string
                        last_action = "Fold"
                    elif action_type.value == 'check':
                        last_action = "Check"
                    elif action_type.value == 'call':
                        amount = getattr(last_action_record.action, 'amount', 0)
                        last_action = f"Call ${amount}" if amount > 0 else "Call"
                    elif action_type.value == 'bet':
                        amount = getattr(last_action_record.action, 'amount', 0)
                        # Check if this is a blind
                        if any('blind' in reason.lower() for reason in last_action_record.action.reasons):
                            blind_type = 'Small Blind' if 'small' in ' '.join(last_action_record.action.reasons).lower() else 'Big Blind'
                            last_action = f"{blind_type} ${amount}"
                        else:
                            last_action = f"Bet ${amount}"
                    elif action_type.value == 'raise':
                        amount = getattr(last_action_record.action, 'amount', 0)
                        last_action = f"Raise to ${amount}"
                
                # Get reasoning if available
                if hasattr(last_action_record.action, 'reasons') and last_action_record.action.reasons:
                    # The reasons should already have emoji prefixes from the agent
                    formatted_reasons = []
                    for reason in last_action_record.action.reasons:
                        reason = reason.strip()
                        if reason:
                            # Ensure proper sentence ending
                            if not reason.endswith('.'):
                                reason += '.'
                            formatted_reasons.append(reason)
                    reasoning = " ".join(formatted_reasons)
            
            hand_str = str(player.hand_name) if player.hand_name is not None else None
            if hand_str and ':' in hand_str:
                hand_str = hand_str.split(':', 1)[0].strip()
            if hand_str and ',' in hand_str:
                hand_str = hand_str.split(',', 1)[0].strip()
            if hand_str and 'Sixs' in hand_str:
                hand_str = hand_str.replace('Sixs', 'Sixes')
            
            players_data.append({
                "name": player.name,
                "chips": player.chips,
                "cards": hole_cards,
                "action": last_action,
                "reasoning": reasoning,
                "hand": hand_str,
                "active": player.is_active,
                "currentBet": player.current_bet,
                "isCurrentPlayer": player.name == (current_player.name if current_player else None)
                #"isCurrentPlayer": player == current_player
                #"isCurrentPlayer": i == current_player_index
            })
        
        # Convert board cards
        board_cards = []
        if hasattr(self.current_hand.game_state, 'board'):
            board_cards = [self._card_to_string(card) for card in self.current_hand.game_state.board]

        return {
            "phase": phase_names.get(self.current_hand.phase, self.current_hand.phase),
            "pot": self.current_hand.game_state.pot,
            "currentBet": self.current_hand.game_state.current_bet,
            "board": board_cards,
            "currentPlayerIndex": current_player_index,
            "currentPlayerName": current_player.name if current_player else None,
            "dealerIndex": self.current_hand.game_state.dealer_position,
            "players": players_data,
            "isComplete": self.current_hand.is_complete,
            'handsPlayed': self.table_manager.hand_no,
            'maxHands': int(getattr(self.config, 'max_hands', 0) or 0)
        }

    def reset_game(self):
        """Reset the entire game"""
        self.setup_players()  # This will recreate the TableManager too
        self.current_hand = None

# Global game instance
poker_game = PokerGameServer()

@app.route('/')
def index():
    """Serve the poker table UI"""
    return render_template('poker_table.html')

@app.route('/api/game/state')
def get_game_state():
    """Get current game state"""
    with game_lock:
        return jsonify(poker_game.get_game_state())

@app.route('/api/game/start', methods=['POST'])
def start_new_hand():
    """Start a new hand"""
    with game_lock:
        logger.info("API: start_new_hand called")
        try:
            state = poker_game.start_new_hand()
            logger.info(f"API: start_new_hand completed, current_hand exists: {poker_game.current_hand is not None}")
            return jsonify(state)
        except Exception as e:
            logger.error(f"API: Error in start_new_hand: {e}", exc_info=True)
            return jsonify({"error": str(e)})

@app.route('/api/game/next', methods=['POST'])
def next_action():
    """Execute next action in the game"""
    with game_lock:
        logger.info(f"API: next_action called, current_hand exists: {poker_game.current_hand is not None}")
        if poker_game.current_hand:
            logger.info(f"API: Hand phase: {poker_game.current_hand.phase}, complete: {poker_game.current_hand.is_complete}")
        try:
            result = poker_game.execute_next_action()
            return jsonify(result)
        except Exception as e:
            logger.error(f"API: Error in next_action: {e}", exc_info=True)
            return jsonify({"error": str(e)})
        
@app.route('/api/game/auto-play', methods=['POST'])
def auto_play():
    """Run the game until completion"""
    with game_lock:
        logger.info("API: auto_play called")
        try:
            result = poker_game.auto_play_until_completion()
            return jsonify(result)
        except Exception as e:
            logger.error(f"API: Error in auto_play: {e}", exc_info=True)
            return jsonify({"error": str(e)})

@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """Reset the entire game"""
    with game_lock:
        poker_game.reset_game()
        return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)