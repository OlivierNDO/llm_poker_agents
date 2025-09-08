"""
Updated mcp_app.py to use individual MCP servers per agent (non-shared)
"""
from flask import Flask, render_template, jsonify, request
import json
import os
import asyncio
import atexit
from threading import Lock
import yaml

from src.game import Hand, Player, TableConfig, StatsTracker, TableManager, Action, ActionType
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
        self.mcp_initialized = False
        self.mcp_enabled = game_cfg.get('mcp_enabled', False)
        self.mcp_script_dir = game_cfg.get('mcp_script_dir', './mcp_servers')
        self.setup_players()
        
    def setup_players(self):
        """Initialize players with individual MCP support."""
        self.players = []
        players_cfg = self.cfg.get('players', [])
        
        for p in players_cfg:
            if not p.get('enabled', True):
                continue
                
            name = str(p.get('name', 'Player'))
            model_id = str(p.get('model', ''))
            chips = int(p.get('chips', 250))
            use_mcp = p.get('use_mcp', False)
            
            client = OpenRouterCompletionEngine(model=model_id)
            
            if use_mcp:
                if not self.mcp_enabled:
                    raise ValueError(f"Player {name} requires MCP but MCP is disabled in config")
                
                # Import individual MCP agent
                try:
                    from mcp_servers.llm_agent_mcp_integration import LLMAgentMCP
                except ImportError as e:
                    raise ImportError(f"MCP agent required for {name} but import failed: {e}")
                
                agent = LLMAgentMCP(
                    name=name,
                    client=client,
                    table_config=self.config,
                    verbose=True,
                    logger=logger
                )
                # Set the MCP script directory after creation
                agent.mcp_script_dir = self.mcp_script_dir
                logger.info(f"Created individual MCP-enabled agent for {name}")
            else:
                # Only allow standard agents if MCP is disabled globally
                if self.mcp_enabled:
                    logger.warning(f"MCP is enabled but {name} not using MCP - consider enabling use_mcp")
                
                # Import the old LLMAgent only if MCP is disabled
                from src.agents import LLMAgent
                agent = LLMAgent(
                    client=client,
                    name=name,
                    table_config=self.config,
                    logger=logger
                )
                logger.info(f"Created standard agent for {name}")
            
            self.players.append(
                Player(
                    name=name,
                    chips=chips,
                    agent=agent
                )
            )
        
        # Validate MCP setup
        if self.mcp_enabled:
            mcp_agents = [p for p in self.players if hasattr(p.agent, 'initialize_mcp')]
            if not mcp_agents:
                raise ValueError("MCP is enabled but no players are using MCP agents")
            logger.info(f"MCP enabled with {len(mcp_agents)} individual MCP agents")
        
        # Create TableManager
        self.table_manager = TableManager(
            players=self.players,
            config=self.config,
            stats_tracker=self.stats_tracker,
            logger=logger
        )

    async def initialize_mcp_servers(self):
        """Initialize individual MCP servers for each agent"""
        if not self.mcp_enabled:
            logger.info("MCP disabled in config, skipping MCP initialization")
            return True
            
        logger.info("Initializing individual MCP servers...")
        
        try:
            # Initialize all MCP agents individually
            mcp_agents = [p for p in self.players if hasattr(p.agent, 'initialize_mcp')]
            
            successful_inits = 0
            failed_agents = []
            
            for player in mcp_agents:
                logger.info(f"Initializing MCP servers for {player.name}...")
                
                try:
                    success = await player.agent.initialize_mcp()
                    if success:
                        successful_inits += 1
                        capabilities = player.agent.get_available_capabilities()
                        logger.info(f"MCP initialized for {player.name}: {', '.join(capabilities)}")
                    else:
                        failed_agents.append(player.name)
                        logger.error(f"MCP initialization failed for {player.name}")
                except Exception as e:
                    failed_agents.append(player.name)
                    logger.error(f"MCP initialization error for {player.name}: {e}")
            
            if failed_agents:
                error_msg = f"MCP initialization failed for agents: {', '.join(failed_agents)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.mcp_initialized = True
            logger.info(f"MCP initialization complete: {successful_inits}/{len(mcp_agents)} agents initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            return False

    async def cleanup_mcp_servers(self):
        """Cleanup individual MCP servers"""
        logger.info("Cleaning up individual MCP servers...")
        
        # Cleanup individual agents
        for player in self.players:
            if hasattr(player.agent, 'cleanup_mcp'):
                try:
                    await player.agent.cleanup_mcp()
                    logger.debug(f"MCP cleanup completed for {player.name}")
                except Exception as e:
                    logger.error(f"MCP cleanup error for {player.name}: {e}")
        
        self.mcp_initialized = False
        logger.info("MCP cleanup completed")

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
            
            # Add MCP status info for debugging/display
            is_mcp_agent = hasattr(player.agent, 'mcp_client')
            mcp_capabilities = []
            if is_mcp_agent and hasattr(player.agent, 'get_available_capabilities'):
                try:
                    mcp_capabilities = player.agent.get_available_capabilities()
                except:
                    pass
            
            players_data.append({
                "name": player.name,
                "chips": player.chips,
                "cards": hole_cards,
                "action": last_action,
                "reasoning": reasoning,
                "hand": hand_str,
                "active": player.is_active,
                "currentBet": player.current_bet,
                "isCurrentPlayer": player.name == (current_player.name if current_player else None),
                "mcpEnabled": is_mcp_agent,
                "mcpCapabilities": mcp_capabilities
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
            'maxHands': int(getattr(self.config, 'max_hands', 0) or 0),
            'mcpEnabled': self.mcp_enabled,
            'mcpInitialized': self.mcp_initialized
        }

    def reset_game(self):
        """Reset the entire game"""
        # Cleanup MCP connections before reset
        if self.mcp_initialized:
            asyncio.run(self.cleanup_mcp_servers())
        
        self.setup_players()  # This will recreate the TableManager too
        self.current_hand = None
        self.mcp_initialized = False

    def get_mcp_status(self):
        """Get MCP status information"""
        status = {
            "enabled": self.mcp_enabled,
            "initialized": self.mcp_initialized,
            "script_dir": self.mcp_script_dir,
            "agents": []
        }
        
        for player in self.players:
            agent_info = {
                "name": player.name,
                "mcp_enabled": hasattr(player.agent, 'mcp_client'),
                "capabilities": []
            }
            
            if hasattr(player.agent, 'get_available_capabilities'):
                try:
                    agent_info["capabilities"] = player.agent.get_available_capabilities()
                except:
                    pass
            
            status["agents"].append(agent_info)
        
        return status

# Global game instance
poker_game = PokerGameServer()

# MCP cleanup on shutdown
async def cleanup_mcp_on_shutdown():
    """Cleanup MCP connections when Flask shuts down"""
    await poker_game.cleanup_mcp_servers()

def shutdown_handler():
    """Handle application shutdown"""
    logger.info("Application shutting down, cleaning up MCP connections...")
    asyncio.run(cleanup_mcp_on_shutdown())

atexit.register(shutdown_handler)

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

@app.route('/api/mcp/status')
def get_mcp_status():
    """Get MCP status information"""
    with game_lock:
        return jsonify(poker_game.get_mcp_status())

@app.route('/api/mcp/initialize', methods=['POST'])
def initialize_mcp():
    """Initialize MCP servers for all agents"""
    with game_lock:
        if not poker_game.mcp_enabled:
            return jsonify({"error": "MCP is disabled in configuration"})
        
        if poker_game.mcp_initialized:
            return jsonify({"status": "already_initialized", "message": "MCP already initialized"})
        
        try:
            # Run MCP initialization
            success = asyncio.run(poker_game.initialize_mcp_servers())
            
            if success:
                return jsonify({
                    "status": "success", 
                    "message": "MCP initialization completed successfully",
                    "mcp_status": poker_game.get_mcp_status()
                })
            else:
                return jsonify({
                    "error": "MCP initialization failed",
                    "status": "failed"
                }), 500
                
        except Exception as e:
            logger.error(f"MCP initialization error: {e}", exc_info=True)
            return jsonify({
                "error": f"MCP initialization failed: {str(e)}",
                "status": "failed"
            }), 500

if __name__ == '__main__':
    # MCP initialization on startup
    if poker_game.mcp_enabled:
        logger.info("MCP enabled, initializing servers...")
        try:
            success = asyncio.run(poker_game.initialize_mcp_servers())
            if not success:
                logger.error("MCP initialization failed")
                exit(1)
            logger.info("MCP initialization completed successfully")
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            logger.error("Cannot start application without working MCP")
            exit(1)
    else:
        logger.info("MCP disabled in configuration")
    
    # Start Flask app
    logger.info("Starting Flask application...")
    app.run(debug=True, port=5000, use_reloader=False)