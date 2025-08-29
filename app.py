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
        self.current_hand = None
        self.stats_tracker = StatsTracker()
        #self.config = TableConfig(small_blind=1, big_blind=2, max_hands=5)
        self.config = TableConfig(
            small_blind=int(game_cfg.get('small_blind', 1)),
            big_blind=int(game_cfg.get('big_blind', 2)),
            max_hands=int(game_cfg.get('max_hands', 5))
        )
        self.players = []
        self.table_manager = None
        self.hand_no = 0
        self.dealer_position = 0  # Track dealer position ourselves
        self.setup_players()
        
        
    def setup_players(self):
        """
        Initialize players from YAML config.
        """
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

    
    def DEPRECATED_setup_players(self):
        """Initialize players with agents"""
        # Create your LLM clients
        
        claude_sonnet_4_client = OpenRouterCompletionEngine(model='anthropic/claude-sonnet-4')
        deepseek_3_1_client = OpenRouterCompletionEngine(model='deepseek/deepseek-chat-v3.1')
        gemini_2_5_client = OpenRouterCompletionEngine(model='google/gemini-2.5-flash')
        grok_4_client = OpenRouterCompletionEngine(model='x-ai/grok-4')
        gpt_5_client = OpenRouterCompletionEngine(model='openai/gpt-5')
        #gpt4_client = OpenRouterCompletionEngine(model='openai/gpt-4o')
        #llama_client = OpenRouterCompletionEngine(model='meta-llama/llama-3.3-70b-instruct')
        #gpt_nano_client = OpenRouterCompletionEngine(model='openai/gpt-4.1-nano')
        #gemma_client = OpenRouterCompletionEngine(model='google/gemma-3-27b-it:free')
        #claude_haiku_client = OpenRouterCompletionEngine(model='anthropic/claude-3-haiku')
        
        
        
        
        # deepseek/deepseek-chat-v3.1
        # anthropic/claude-sonnet-4
        # google/gemma-3-27b-it:free
        # openai/gpt-4.1-nano
        
        
        
        
        self.players = [
            
            
            Player(
                name='Claude Sonnet 4',
                chips=250,
                agent=LLMAgent(
                    client=claude_sonnet_4_client,
                    name='Claude Sonnet 4',
                    table_config=self.config,
                    logger = logger
                )
            ),
            
            Player(
                name='DeepSeek 3.1',
                chips=250,
                agent=LLMAgent(
                    client=deepseek_3_1_client,
                    name='DeepSeek 3.1',
                    table_config=self.config,
                    logger = logger
                )
            ),
            
            Player(
                name='Gemini 2.5 Flash',
                chips=250,
                agent=LLMAgent(
                    client=gemini_2_5_client,
                    name='Gemini 2.5 Flash',
                    table_config=self.config,
                    logger = logger
                )
            ),
            
            #Player(
            #    name='Grok 4',
            #    chips=250,
            #    agent=LLMAgent(
            #        client=grok_4_client,
            #        name='Grok 4',
            #        table_config=self.config,
            #        logger = logger
            #    )
            #),
            
            Player(
                name='Gpt 5',
                chips=250,
                agent=LLMAgent(
                    client=gpt_5_client,
                    name='Gpt 5',
                    table_config=self.config,
                    logger = logger
                )
            ),
            
            
            

            
            #Player(
            #    name='Gemma 3 27B',
            #    chips=250,
            #    agent=LLMAgent(
            #        client=gemma_client,
            #        name='Gemma 3 27B', 
            #        table_config=self.config,
            #        logger = logger
            #    )
            #),
            
        ]

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
    
    def _get_active_players_in_order(self):
        """Get players with chips > 0 in proper seating order starting from dealer"""
        funded_players = [p for p in self.players if p.chips > 0]
        if len(funded_players) <= 1:
            return funded_players
            
        # For heads-up, keep original order but track dealer
        if len(funded_players) == 2:
            return funded_players
            
        # For 3+ players, order starting from dealer position
        ordered = []
        n = len(self.players)
        for offset in range(n):
            idx = (self.dealer_position + offset) % n
            if self.players[idx].chips > 0:
                ordered.append(self.players[idx])
        return ordered
    
    def DEPRECATED_get_dealer_position_in_hand(self, hand_players):
        """Get dealer position within the hand's player list"""
        if len(hand_players) == 2:
            # For heads-up, find which position the dealer is in
            for i, player in enumerate(hand_players):
                if self.players[self.dealer_position] is player:
                    return i
            return 0
        else:
            # For 3+, dealer is always at position 0 in our ordering
            return 0
        
    def _get_dealer_position_in_hand(self, hand_players):
        """Get dealer position within the hand's player list"""
        if len(hand_players) == 2:
            # For heads-up, find which position the actual dealer occupies
            dealer_player = self.players[self.dealer_position]
            for i, player in enumerate(hand_players):
                if player is dealer_player:
                    return i
            return 0  # fallback
        else:
            # For 3+ players, dealer is always at position 0 due to ordering
            return 0
    
    def _advance_dealer_button(self):
        """Move dealer button to next funded player"""
        original_dealer = self.dealer_position
        n = len(self.players)
        
        # Find next player with chips
        for i in range(1, n + 1):
            next_idx = (self.dealer_position + i) % n
            if self.players[next_idx].chips > 0:
                self.dealer_position = next_idx
                break
        
        print(f"Dealer button moved from position {original_dealer} to {self.dealer_position}")
    
    def _post_blinds(self, hand):
        """Post blinds similar to TableManager._post_blinds"""
        logger.info("Starting blind posting process")
        
        gs = hand.game_state
        players = gs.players
        n = len(players)
        
        if n < 2:
            logger.warning("Not enough players to post blinds")
            return
            
        dealer_pos = gs.dealer_position
        logger.info(f"Dealer position: {dealer_pos}, Total players: {n}")
        
        if n == 2:
            # Heads-up: Dealer posts SB, other posts BB
            sb_index = dealer_pos
            bb_index = 1 - dealer_pos
            first_to_act = dealer_pos
        else:
            # 3+ players
            sb_index = (dealer_pos + 1) % n
            bb_index = (dealer_pos + 2) % n 
            first_to_act = (dealer_pos + 3) % n if n > 3 else dealer_pos
        
        logger.info(f"SB index: {sb_index}, BB index: {bb_index}, First to act: {first_to_act}")
        
        try:
            # Post small blind
            sb_player = players[sb_index]
            sb_amt = min(self.config.small_blind, sb_player.chips)
            logger.info(f"Posting small blind: {sb_player.name} posts {sb_amt}")
            sb_action = Action(ActionType.BET, sb_amt, reasons=["Small blind required"])
            gs.apply_action(sb_player, sb_action)
            
            # Post big blind  
            bb_player = players[bb_index]
            bb_amt = min(self.config.big_blind, bb_player.chips)
            logger.info(f"Posting big blind: {bb_player.name} posts {bb_amt}")
            bb_action = Action(ActionType.BET, bb_amt, reasons=["Big blind required"])
            gs.apply_action(bb_player, bb_action)
            
            # Set game state
            gs.current_bet = bb_amt
            gs.current_player_index = first_to_act
            gs.betting_round = 0
            
            logger.info(f"Blinds posted successfully. Current bet: {bb_amt}, First to act: {players[first_to_act].name}")
            
        except Exception as e:
            logger.error(f"Error posting blinds: {e}", exc_info=True)
            raise
    
    def start_new_hand(self):
        """Start a new hand with proper dealer rotation"""
        # 8/29
        if getattr(self.config, 'max_hands', 0) and self.hand_no >= int(self.config.max_hands):
            return {
                'error': 'Max hands reached',
                'complete': True,
                'handsPlayed': self.hand_no,
                'maxHands': int(self.config.max_hands),
                'game_state': self.get_game_state() if self.current_hand else {}
            }
        logger.info(f"Starting new hand #{self.hand_no + 1}")
        
        # Get active players in correct seating order
        active_players = self._get_active_players_in_order()
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
        
        # Set dealer position in hand
        dealer_in_hand = self._get_dealer_position_in_hand(active_players)
        self.current_hand.game_state.dealer_position = dealer_in_hand
        logger.info(f"Dealer position in hand: {dealer_in_hand}")
        
        # Execute setup and deal hole cards
        logger.info("Executing hand setup...")
        self.current_hand.execute_next_step()  # setup
        logger.info("Dealing hole cards...")
        self.current_hand.execute_next_step()  # deal hole cards
        
        # Post blinds
        logger.info("Posting blinds...")
        self._post_blinds(self.current_hand)
        
        # Advance to preflop phase
        self.current_hand.phase = "preflop"
        logger.info("Hand setup complete, phase set to preflop")
        
        self.hand_no += 1
        
        return self.get_game_state()


    def auto_play_until_completion(self):
        """Run the game until completion (max hands or one winner)"""
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
                    logger.info(f"Completing current hand (hand #{self.hand_no})")
                    
                    while not self.current_hand.is_complete:
                        step_result = self.current_hand.execute_next_step()
                        if step_result:
                            log_entries.append(f"Hand #{self.hand_no}: {step_result}")
                    
                    # Advance dealer button after hand completion
                    if self.current_hand.is_complete:
                        self._advance_dealer_button()
                        log_entries.append(f"Hand #{self.hand_no} complete")
                
                # Check max hands limit BEFORE starting next hand
                if hasattr(self.config, 'max_hands') and self.config.max_hands > 0:
                    if self.hand_no >= self.config.max_hands:
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
                logger.info(f"Starting hand #{self.hand_no + 1}")
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
                log_entries.append(f"Started hand #{self.hand_no}")
            
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
                self._advance_dealer_button()
            
            return {
                "step_result": result,
                "game_state": self.get_game_state(),
                "complete": self.current_hand.is_complete
            }
        except Exception as e:
            logger.error(f"Error executing next action: {e}", exc_info=True)
            return {"error": f"Failed to execute action: {str(e)}"}
    
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
                last_action = last_action_record.action.action_type.value
                if last_action_record.action.amount > 0:
                    last_action += f" ${last_action_record.action.amount}"
                
                # Get reasoning if available
                if hasattr(last_action_record.action, 'reasons') and last_action_record.action.reasons:
                    #reasoning = "; ".join(last_action_record.action.reasons)
                    #reasoning = ". ".join(last_action_record.action.reasons)
                    # Clean up each reason and join properly
                    cleaned_reasons = []
                    for reason in last_action_record.action.reasons:
                        reason = reason.strip()
                        if reason and not reason.endswith('.'):
                            reason += '.'
                        cleaned_reasons.append(reason)
                    reasoning = " ".join(cleaned_reasons)
            
            
            hand_str = str(player.hand_name) if player.hand_name is not None else None
            if hand_str and ':' in hand_str:
                hand_str = hand_str.split(':', 1)[0].strip()
            if hand_str and ',' in hand_str:
                hand_str = hand_str.split(',', 1)[0].strip()
            
            players_data.append({
                "name": player.name,
                "chips": player.chips,
                "cards": hole_cards,
                "action": last_action,
                "reasoning": reasoning,
                "hand": hand_str,
                "active": player.is_active,
                "currentBet": player.current_bet,
                "isCurrentPlayer": i == current_player_index
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
            # 8/29
            'handsPlayed': self.hand_no,
            'maxHands': int(getattr(self.config, 'max_hands', 0) or 0)
        }

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
        poker_game.setup_players()
        poker_game.dealer_position = 0
        poker_game.hand_no = 0
        poker_game.current_hand = None
        return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)