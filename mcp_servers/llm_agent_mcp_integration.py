"""
llm_agent_mcp_integration.py

Monte Carlo only version - removes TensorFlow ML model complexity
"""

import asyncio
import json
import os
from typing import List, Optional, Dict, Any

from src.poker_types import ActionType, Action
from src.agent_utils import OpenRouterCompletionEngine, card_to_str
from src.logging_config import logger
import src.feature_engineering as fe

# Import MCP client
try:
    from mcp_servers.mcp_poker_client import MCPPokerClient
except ImportError:
    print("Warning: MCPPokerClient not found, MCP will be disabled")
    class MCPPokerClient:
        def get_available_capabilities(self):
            return []

# Import your existing classes - type checking only
if False:
    from src.game import GameState, Player, StatsTracker, TableConfig


class LLMAgentMCP:
    """
    MCP-enabled version of LLMAgent that uses MCP servers for analysis.
    
    Uses MCP servers for:
    - Hand Analysis: Mathematical analysis, probabilities, hand strength
    - Bet Sizing: Context-aware sizing recommendations  
    - Game State: Formatted game information, pot odds, position analysis
    - Stats Tracker: Opponent analysis and behavioral patterns
    """

    def __init__(
        self,
        name: str = 'LLMAgentMCP',
        player_profile: Optional[str] = None,
        client=None,
        table_config: Optional['TableConfig'] = None,
        verbose: bool | int = True,
        log_prompts: bool = False,
        logger=logger
    ):
        self.name = name
        self.player_profile = player_profile or (
            'You are an AI poker agent playing Texas Hold\'em against other agents. '
            'Make rational decisions based on pot odds, position, and opponent tendencies. '
        )
        self.llm_client = client or OpenRouterCompletionEngine(
            model='openai/gpt-4o',
            token=os.getenv('OPEN_ROUTER_TOKEN'),
            url='https://openrouter.ai/api/v1/chat/completions'
        )
        self.table_config = table_config
        self.verbose = bool(verbose)
        self.log_prompts = log_prompts
        self.logger = logger
        
        # MCP configuration - will be set by the Flask app
        self.mcp_script_dir = './mcp_servers'
        self.mcp_client = None
        self.available_capabilities = []

    async def initialize_mcp(self) -> bool:
        """Initialize all available MCP servers"""
        try:
            # Import MCP client
            self.mcp_client = MCPPokerClient()
            
            # Connect to hand analysis server
            script_path = os.path.join(self.mcp_script_dir, "hand_analysis_server.py")
            hand_success = await self.mcp_client.connect_to_hand_analysis_server(script_path)
            
            # Connect to Monte Carlo server
            script_path = os.path.join(self.mcp_script_dir, "predictive_model_server.py")
            pred_success = await self.mcp_client.connect_to_predictive_model_server(script_path)
            
            # Consider initialization successful if at least one server connects
            if hand_success or pred_success:
                self.available_capabilities = self.mcp_client.get_available_capabilities()
                if self.verbose:
                    status = []
                    if hand_success:
                        status.append("hand_analysis")
                    if pred_success:
                        status.append("monte_carlo")
                    self.logger.info(f"{self.name}: Connected to MCP servers: {status}. Capabilities: {self.available_capabilities}")
                return True
            else:
                if self.verbose:
                    self.logger.warning(f"{self.name}: Failed to connect to any MCP servers")
                
        except Exception as e:
            if self.verbose:
                self.logger.error(f"{self.name}: MCP initialization failed: {e}")
            return False

    async def cleanup_mcp(self):
        """Cleanup MCP connections"""
        if self.mcp_client:
            await self.mcp_client.disconnect_all()
            self.mcp_client = None
            self.available_capabilities = []

    def get_available_capabilities(self) -> List[str]:
        """Get list of available MCP capabilities"""
        return self.available_capabilities.copy()

    def get_action(
        self,
        player: 'Player',
        game_state: 'GameState',
        valid_actions: List[ActionType],
        stats_tracker: Optional['StatsTracker'] = None,
        monte_carlo_tracker = None  # ADD THIS PARAMETER
    ) -> Action:
        """Synchronous wrapper for async get_action_async"""
        return asyncio.run(self.get_action_async(player, game_state, valid_actions, stats_tracker, monte_carlo_tracker))

    async def get_action_async(
        self,
        player: 'Player',
        game_state: 'GameState',
        valid_actions: List[ActionType],
        stats_tracker: Optional['StatsTracker'] = None,
        monte_carlo_tracker = None
    ) -> Action:
        """Complete MCP-enabled decision making process with proper data isolation"""
        
        # FORCE INITIALIZE - Set attributes directly, don't rely on clearing
        player.hand_analysis = ""
        player.opponent_stats = ""
        player.monte_carlo_result = ""
        player.planning_reasoning = ""
        
        if self.verbose:
            self.logger.info(f"{self.name}: Starting decision process with clean slate")
        
        # Ensure MCP is initialized
        if not self.mcp_client:
            await self.initialize_mcp()
    
        # Three-stage process
        planning_decision = await self._get_planning_decision_mcp(player, game_state, valid_actions, monte_carlo_tracker)
        gathered_info = await self._gather_information_mcp(planning_decision, player, game_state, stats_tracker, monte_carlo_tracker)
        final_action = await self._make_betting_decision_mcp(player, game_state, valid_actions, gathered_info)
        
        # Store planning reasoning
        planning_reasons = planning_decision.get("reasoning", [])
        utilities_lines = ["Utilities Called:"]
        if planning_decision.get('need_hand_analysis', False):
            utilities_lines.append("• Hand Analysis")
        if planning_decision.get('need_outcome_prediction', False):
            utilities_lines.append("• Monte Carlo")
        if planning_decision.get('need_opponent_stats', False):
            utilities_lines.append("• Opponent Stats")
        
        if len(utilities_lines) == 1:
            utilities_lines.append("• None")
        
        utilities_summary = "<br>".join(utilities_lines)
        planning_parts = [utilities_summary]
        if planning_reasons:
            planning_parts.extend(planning_reasons)
        
        player.planning_reasoning = ". ".join(planning_parts) if planning_parts else "No planning provided."
        
        # FINAL SAFETY CHECK - Log what we're ending up with
        #if self.verbose:
        self.logger.info(f"{self.name}: FINAL MCP DATA CHECK:")
        self.logger.info(f"  hand_analysis: {repr(getattr(player, 'hand_analysis', 'MISSING'))}")
        self.logger.info(f"  opponent_stats: {repr(getattr(player, 'opponent_stats', 'MISSING'))}")
        self.logger.info(f"  monte_carlo_result: {repr(getattr(player, 'monte_carlo_result', 'MISSING'))}")
        
        return final_action


    async def _get_planning_decision_mcp(
        self,
        player: 'Player',
        game_state: 'GameState',
        valid_actions: List[ActionType],
        monte_carlo_tracker = None  # ADD THIS PARAMETER
    ) -> dict:
        """MCP-enabled planning with usage tracking"""
        
        # Get basic game info
        basic_info = self._get_basic_prompt_info_fallback(player, game_state)
        legal_str = ", ".join(a.value for a in valid_actions)
        
        # Check Monte Carlo availability
        monte_carlo_available = False
        usage_status = "Monte Carlo: Not available"
        
        if monte_carlo_tracker and "monte_carlo_simulation" in self.available_capabilities:
            monte_carlo_available = monte_carlo_tracker.can_use_monte_carlo(player.name)
            usage_status = monte_carlo_tracker.get_usage_status(player.name)
        
        # Dynamic capability listing with usage info
        capability_descriptions = {
            "comprehensive_hand_analysis": "HAND ANALYSIS: Mathematical strength, probabilities, equity calculations",
            "monte_carlo_simulation": f"MONTE CARLO: Outcome probabilities via simulation (2,500 iterations) - {usage_status}",
            "quick_win_probability": f"WIN PROBABILITY: Fast win chance estimate (1,000 iterations) - {usage_status}",
        }
        
        capability_list = []
        for capability in self.available_capabilities:
            # Only include Monte Carlo capabilities if available
            if capability in ["monte_carlo_simulation", "quick_win_probability"]:
                if monte_carlo_available:
                    description = capability_descriptions.get(capability, f"ANALYSIS: {capability}")
                    capability_list.append(description)
                # Skip if Monte Carlo is used up
            else:
                description = capability_descriptions.get(capability, f"ANALYSIS: {capability}")
                capability_list.append(description)
        
        if not capability_list:
            capability_list = ["LIMITED ANALYSIS: Basic game state only"]
        
        capabilities_text = "\n".join(f"{i+1}. {cap}" for i, cap in enumerate(capability_list))
        
        
        planning_prompt = f"""
        PROFILE: {self.player_profile}
        
        TASK: Plan your poker decision. You have access to analytical tools - use them strategically.
        
        SITUATION:
        {basic_info}
        
        LEGAL ACTIONS: [{legal_str}]
        
        AVAILABLE ANALYSIS CAPABILITIES:
        {capabilities_text}
        
        ANALYSIS BUDGET: {usage_status}
        IMPORTANT CONSIDERATIONS:
            > Monte Carlo analysis can only be used ONCE per hand. Choose when to use it wisely. Only in rare occasions would it be wise to use this pre-flop.
            > Opponent stats can be used any time, but depending on the number of hands played, they may or may not be informative.
            
        You must output exactly this JSON format:
        {{"need_hand_analysis": true, "need_outcome_prediction": {str(monte_carlo_available).lower()}, "need_opponent_stats": true, "ready_to_decide": false, "reasoning": ["short reason 1", "short reason 2"]}}
        
        DECISION FRAMEWORK:
        - need_hand_analysis: Consider if hand analysis would help your decision
        - need_outcome_prediction: {str(monte_carlo_available).upper()} (Monte Carlo {'available - USE WISELY' if monte_carlo_available else 'exhausted for this hand'})
        - need_opponent_stats: Consider if opponent reads would influence your decision  
        - ready_to_decide: Only TRUE if confident without additional analysis
        - reasoning: 1-3 reasons

        Output JSON only:"""
    
        if self.log_prompts:
            self.logger.info('Planning prompt submitted:')
            self.logger.info(planning_prompt)
            
        raw_text = self.llm_client.submit_prompt_return_response(planning_prompt)
        raw_text = (raw_text or '').strip()
        raw_text = raw_text.replace("```json", "").replace("```", "")
        planning = self._parse_planning_json(raw_text)
        
        if self.verbose:
            reasons = " | ".join(planning.get("reasoning", ["No reasoning provided"]))
            self.logger.info(f"{self.name} Planning: hand_analysis={planning.get('need_hand_analysis', False)}, "
               f"outcome_prediction={planning.get('need_outcome_prediction', False)}, "
               f"opponent_stats={planning.get('need_opponent_stats', False)}, "
               f"ready={planning.get('ready_to_decide', True)} | {usage_status} | {reasons}")
        
        return planning

    async def _gather_information_mcp(
        self,
        planning_decision: dict,
        player: 'Player',
        game_state: 'GameState',
        stats_tracker: Optional['StatsTracker'],
        monte_carlo_tracker = None
    ) -> dict:
        """MCP-enabled information gathering with proper data isolation"""
        
        gathered = {}
        
        # Hand analysis via MCP - ONLY if requested
        if planning_decision.get("need_hand_analysis", False) and self.mcp_client:
            try:
                hole_cards = [card_to_str(c) for c in player.hole_cards] if player.hole_cards else []
                board_cards = [card_to_str(c) for c in game_state.board] if game_state.board else []
                num_opponents = len(game_state.get_active_players()) - 1
                
                if "comprehensive_hand_analysis" in self.available_capabilities:
                    analysis = await self.mcp_client.get_hand_analysis(hole_cards, board_cards, num_opponents)
                    if analysis and analysis.strip():  # Only store if we got real data
                        gathered["hand_analysis"] = analysis
                        player.hand_analysis = analysis
                        
                        if self.verbose:
                            self.logger.info(f"{self.name}: HAND ANALYSIS SUCCESS:")
                            self.logger.info(f"  Cards: {hole_cards} | Board: {board_cards} | Opponents: {num_opponents}")
                            self.logger.info(f"  Analysis: {analysis[:100]}...")
                    else:
                        if self.verbose:
                            self.logger.warning(f"{self.name}: Hand analysis returned empty result")
                
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"{self.name}: Hand analysis MCP failed: {e}")
    
        # Opponent stats via direct StatsTracker - ONLY if requested
        if planning_decision.get("need_opponent_stats", False) and stats_tracker:
            try:
                opponent_stats = stats_tracker.get_opponent_stats_summary(player.name)
                if opponent_stats and opponent_stats.strip():  # Only store if we got real data
                    gathered["opponent_stats"] = opponent_stats
                    player.opponent_stats = opponent_stats
                    
                    if self.verbose:
                        self.logger.info(f"{self.name}: OPPONENT STATS SUCCESS:")
                        self.logger.info(f"  Stats: {opponent_stats[:100]}...")
                else:
                    if self.verbose:
                        self.logger.warning(f"{self.name}: Opponent stats returned empty result")
                        
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"{self.name}: Opponent stats gathering failed: {e}")
    
        # Monte Carlo predictions - ONLY if requested and available
        if planning_decision.get("need_outcome_prediction", False) and self.mcp_client and monte_carlo_tracker:
            if monte_carlo_tracker.can_use_monte_carlo(player.name):
                try:
                    hole_cards = [card_to_str(c) for c in player.hole_cards] if player.hole_cards else []
                    board_cards = [card_to_str(c) for c in game_state.board] if game_state.board else []
                    num_players = len(game_state.get_active_players())
                    
                    if "monte_carlo_simulation" in self.available_capabilities:
                        prediction = await self.mcp_client.predict_outcome_monte_carlo(hole_cards, board_cards, num_players, 2500)
                        
                        # RECORD USAGE AFTER SUCCESSFUL CALL
                        monte_carlo_tracker.record_usage(player.name)
                        
                        if prediction:
                            monte_carlo_text = f"Monte Carlo Prediction (2,500 simulations) - Win: {prediction['win_percentage']}, Tie: {prediction['tie_percentage']}, Loss: {prediction['loss_percentage']} | Samples: {prediction['samples']}"
                            gathered["outcome_prediction"] = monte_carlo_text
                            player.monte_carlo_result = monte_carlo_text
                            
                            if self.verbose:
                                self.logger.info(f"{self.name}: MONTE CARLO SUCCESS:")
                                self.logger.info(f"  Cards: {hole_cards} | Board: {board_cards} | Players: {num_players}")
                                self.logger.info(f"  Results: {monte_carlo_text}")
                                
                    elif "quick_win_probability" in self.available_capabilities:
                        win_prob = await self.mcp_client.get_win_probability_fast(hole_cards, board_cards, num_players)
                        
                        # RECORD USAGE AFTER SUCCESSFUL CALL
                        monte_carlo_tracker.record_usage(player.name)
                        
                        if win_prob:
                            win_prob_text = f"Quick Win Probability: {win_prob:.1%} (1,000 Monte Carlo simulations)"
                            gathered["outcome_prediction"] = win_prob_text
                            player.monte_carlo_result = win_prob_text
                            
                            if self.verbose:
                                self.logger.info(f"{self.name}: QUICK WIN PROBABILITY SUCCESS: {win_prob_text}")
                    
                except Exception as e:
                    if self.verbose:
                        self.logger.error(f"{self.name}: Monte Carlo prediction failed: {e}")
            else:
                if self.verbose:
                    self.logger.info(f"{self.name}: Monte Carlo skipped - {monte_carlo_tracker.get_usage_status(player.name)}")
    
        return gathered

    async def _make_betting_decision_mcp(
        self,
        player: 'Player',
        game_state: 'GameState',
        valid_actions: List[ActionType],
        gathered_info: dict
    ) -> Action:
        """Make betting decision using gathered MCP information"""
        
        # Build comprehensive information context
        info_sections = []
        
        # ALWAYS include basic game info first (hole cards, chips, pot, etc.)
        info_sections.append(self._get_basic_prompt_info_fallback(player, game_state))
        
        # Add hand analysis if available
        if "hand_analysis" in gathered_info:
            info_sections.append(f"DETAILED HAND ANALYSIS:\n{gathered_info['hand_analysis']}")
        
        # Add opponent statistics if available
        if "opponent_stats" in gathered_info:
            info_sections.append(f"OPPONENT STATISTICS:\n{gathered_info['opponent_stats']}")
            
        # Add outcome predictions if available
        if "outcome_prediction" in gathered_info:
            info_sections.append(f"OUTCOME PREDICTION:\n{gathered_info['outcome_prediction']}")
        
        combined_info = "\n\n".join(info_sections)
        legal_str = ", ".join(a.value for a in valid_actions)
        
        # Enhanced decision prompt
        decision_prompt = f"""{self.player_profile}

        COMPREHENSIVE POKER ANALYSIS:
        {combined_info}
        
        LEGAL ACTIONS: [{legal_str}]
        
        Based on this analysis, choose your optimal action.
        
        Your response must be valid JSON only:
        {{"action": "fold|check|call|bet|raise|all_in", "amount": null_or_integer, "reasons": ["reason1", "reason2"]}}
        
        REASONING REQUIREMENTS:
        - Maximum 2-3 reasons
        - Be concise and direct
        
        Key decision factors:
        - Hand strength and drawing potential
        - Pot odds and mathematical correctness
        - Opponent tendencies and likely holdings
        - Position and betting round dynamics
        - Exploitative opportunities and opponent image management
        
        Your response must start with {{ and contain only valid JSON:"""

        if self.log_prompts:
            self.logger.info('Decision prompt submitted')
            
        raw_response = self.llm_client.submit_prompt_return_response(decision_prompt)
        raw_response = (raw_response or '').strip()
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        
        try:
            decision = json.loads(raw_response)
            return self._validate_and_create_action_mcp(decision, valid_actions, game_state, player)
        except Exception as e:
            if self.verbose:
                self.logger.error(f"{self.name}: Decision parsing failed: {e}")
            # Fallback to safe action
            if ActionType.FOLD in valid_actions:
                return Action(ActionType.FOLD)
            elif ActionType.CHECK in valid_actions:
                return Action(ActionType.CHECK)
            else:
                return Action(valid_actions[0])

    def _get_basic_prompt_info_fallback(self, player: 'Player', game_state: 'GameState') -> str:
        """Fallback basic info when MCP is unavailable"""
        info = player.get_hand_info(game_state)
        
        lines = []
        lines.append(f"You are {player.name}, competing in a Texas Hold'em hand.")
        lines.append(f"Your hole cards: {info['my_hole_cards'][0]} {info['my_hole_cards'][1]}")
        lines.append(f"Your chips: {info['my_chips']}")
        lines.append(f"Current pot: {info['pot']}")
        
        if info['board']:
            board_str = " ".join(str(card) for card in info['board'])
            lines.append(f"Community cards: {board_str}")
        else:
            lines.append("Community cards: None dealt yet")
            
        lines.append(f"Betting round: {info['betting_round_name']}")
        
        if info['amount_to_call'] > 0:
            lines.append(f"You need to call {info['amount_to_call']} to stay in the hand")
        else:
            lines.append("No bet to call - you can check or bet")
        
        return "\n".join(lines)

    def _parse_planning_json(self, raw_text: str) -> dict:
        """Parse planning JSON with robust error handling"""
        try:
            return json.loads(raw_text)
        except:
            # Try to extract JSON from text
            import re
            json_match = re.search(r'\{[^}]*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # Fallback to conservative planning
            if self.verbose:
                self.logger.warning(f"{self.name}: Planning JSON parsing failed, using fallback")
            
            return {
                "need_hand_analysis": True,
                "need_opponent_stats": True,
                "need_outcome_prediction": True,
                "ready_to_decide": False,
                "reasoning": ["JSON parse failed, gathering all available info"]
            }

    def _validate_and_create_action_mcp(
        self,
        decision: dict,
        valid_actions: List[ActionType],
        game_state: 'GameState',
        player: 'Player'
    ) -> Action:
        """Validate LLM decision and create Action object"""
        
        # DEBUG: Log the current game situation
        if self.verbose:
            self.logger.info(f"VALIDATION DEBUG for {player.name}:")
            self.logger.info(f"  Player chips: {player.chips}")
            self.logger.info(f"  Player current bet: {player.current_bet}")
            self.logger.info(f"  Game current bet: {game_state.current_bet}")
            self.logger.info(f"  Pot: {game_state.pot}")
            self.logger.info(f"  Valid actions: {[a.value for a in valid_actions]}")
            self.logger.info(f"  Player decision: {decision}")
            
        
        if not isinstance(decision, dict):
            raise TypeError('LLM JSON must be an object')
        
        action_str = decision.get('action')
        if not isinstance(action_str, str):
            raise ValueError('Missing or invalid "action"')
        
        # Create action type mapping to handle case issues
        action_mapping = {
            'fold': ActionType.FOLD,
            'check': ActionType.CHECK,
            'call': ActionType.CALL,
            'bet': ActionType.BET,
            'raise': ActionType.RAISE,
            'all_in': ActionType.ALL_IN
        }
        
        action_str_lower = action_str.lower().strip()
        atype = action_mapping.get(action_str_lower)
        
        if atype is None:
            raise ValueError(f'Unknown action "{action_str}"')
        
        if atype not in valid_actions:
            # Debug logging to help diagnose the issue
            valid_action_names = [a.value for a in valid_actions]
            if self.verbose:
                self.logger.error(f'Action validation failed: {atype.value} not in {valid_action_names}')
            raise ValueError(f'Illegal action: {atype.value} not in {valid_action_names}')
        
        # Handle amount validation
        amount_field = decision.get('amount', None)
        amt = 0
        
        if atype in (ActionType.BET, ActionType.RAISE):
            if amount_field is None:
                raise ValueError('amount must be provided for bet/raise')
            if not isinstance(amount_field, int):
                raise TypeError('amount must be an integer for bet/raise')
            if amount_field < 1:
                raise ValueError('amount must be >= 1 for bet/raise')
            
            # Calculate max affordable
            call_needed = max(0, game_state.current_bet - player.current_bet)
            max_bet = player.chips
            max_raise = player.chips - call_needed
            
            max_afford = max_bet if atype == ActionType.BET else max_raise
            if max_afford <= 0:
                raise ValueError('Insufficient chips for requested bet/raise')
            if amount_field > max_afford:
                raise ValueError(f'amount {amount_field} exceeds max affordable {max_afford}')
            
            amt = amount_field
            
        elif atype == ActionType.CALL:
            amt = max(0, game_state.current_bet - player.current_bet)
        elif atype == ActionType.ALL_IN:
            amt = player.chips
        else:
            if amount_field not in (None, 0):
                raise ValueError(f'Non-bet/raise action must not include amount, got {amount_field}')
        
        # Extract reasons
        reasons = decision.get('reasons', None)
        if reasons is not None:
            if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
                raise TypeError('"reasons" must be a list of strings')
            reasons = [r.strip() for r in reasons if r and isinstance(r, str)]
        
        if self.verbose:
            amount_phrase = f" {amt}" if atype in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN) else ""
            reason_text = "; ".join(reasons) if reasons else "no reasons provided"
            self.logger.info(f"{self.name}: Final decision: {atype.value}{amount_phrase} because {reason_text}")
        
        return Action(atype, amt, reasons=reasons)