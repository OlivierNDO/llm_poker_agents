"""
agents.py
"""


import random
from typing import List, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv
import json
import os
import numpy as np
import nlpcloud
import warnings
import requests
import json

from src.agent_utils import BetSizingAnalyzer, FeatureReporter, OpenRouterCompletionEngine
from src.core_poker_mechanics import Card, Suit, HandEvaluator
import src.feature_engineering as fe

from src.game import Action, ActionType, GameState, Player, TableConfig, StatsTracker

from src.logging_config import logger



class Agent(ABC):
    @abstractmethod
    def get_action(self, player: Player, game_state: GameState, valid_actions: List[ActionType]) -> Action:
        pass
    
    
    
class AnalyticalAgent(Agent):
    """Example agent that uses action history to make decisions"""
    def get_action(self, player: Player, game_state: GameState, valid_actions: List[ActionType]) -> Action:
        # Example: Look at recent actions to inform decision
        recent_actions = game_state.get_actions_since_board_change()
        
        # Count aggressive actions this round
        aggressive_actions = sum(1 for a in recent_actions 
                               if a.action.action_type in [ActionType.BET, ActionType.RAISE])
        
        # Be more conservative if lots of aggression
        if aggressive_actions >= 2:
            if ActionType.FOLD in valid_actions:
                return Action(ActionType.FOLD)
            elif ActionType.CHECK in valid_actions:
                return Action(ActionType.CHECK)
            elif ActionType.CALL in valid_actions:
                return Action(ActionType.CALL)
        
        # Look at specific opponent behavior
        opponent_actions = []
        for other_player in game_state.get_active_players():
            if other_player != player:
                opponent_actions.extend(game_state.get_player_actions(other_player.name))
        
        # Simple logic - if opponents have been passive, be more aggressive
        if not opponent_actions or all(a.action.action_type in [ActionType.CHECK, ActionType.CALL] 
                                     for a in opponent_actions[-3:]):  # Last 3 actions
            if ActionType.BET in valid_actions:
                return Action(ActionType.BET, min(5, player.chips))
            elif ActionType.RAISE in valid_actions:
                return Action(ActionType.RAISE, min(5, player.chips - (game_state.current_bet - player.current_bet)))
        
        # Default to random
        return RandomAgent().get_action(player, game_state, valid_actions)




class LLMAgent:
    """
    Enhanced poker agent that uses a two-stage process:
    1. Planning stage - decides what information to gather
    2. Decision stage - makes the actual betting decision
    
    Expects planning JSON like:
    {
      "need_opponent_stats": true,
      "need_hand_analysis": false,
      "ready_to_decide": false,
      "reasoning": ["opponent has been aggressive", "need to understand their patterns"]
    }
    
    And decision JSON like:
    {
      "action": "raise",
      "amount": 12,
      "reasons": ["strong hand", "good position"]
    }
    """

    def __init__(self, name: str = 'LLMAgent', player_profile: Optional[str] = None,
                 client = OpenRouterCompletionEngine(
                     model='openai/gpt-4o',
                     token=os.getenv('OPEN_ROUTER_TOKEN'),
                     url='https://openrouter.ai/api/v1/chat/completions'
                 ),
                 table_config: Optional[TableConfig] = None,
                 verbose: bool | int = True,
                 log_prompts: bool = False,
                 logger = logger):
        self.name = name
        self.player_profile = player_profile or (
            'You are an AI poker agent playing Texas Hold\'em against other agents. '
            'Make rational decisions based on pot odds, position, and opponent tendencies. '
        )
        self.client = client
        self.table_config = table_config
        self.verbose = bool(verbose)
        self.log_prompts = log_prompts
        self.logger = logger

    def get_action(
        self,
        player: Player,
        game_state: GameState,
        valid_actions: List[ActionType],
        stats_tracker: Optional[StatsTracker] = None
    ) -> Action:
        """
        Two-stage process: Planning → Information Gathering → Decision
        """
        planning_decision = self._get_planning_decision(player, game_state, valid_actions)
        gathered_info = self._gather_information(planning_decision, player, game_state, stats_tracker)
        return self._make_betting_decision(player, game_state, valid_actions, gathered_info)

    def _get_planning_decision(
        self,
        player: Player,
        game_state: GameState,
        valid_actions: List[ActionType]
    ) -> dict:
        """
        First stage: Ask the LLM what information it needs before deciding
        """
        basic_info = self._get_basic_prompt_info(player, game_state)
        legal_str = ", ".join(a.value for a in valid_actions)
        
        planning_prompt = f"""PROFILE: {self.player_profile}

TASK: Plan your poker decision. You have access to powerful analytical tools - use them strategically.

SITUATION:
{basic_info}

LEGAL ACTIONS: [{legal_str}]

AVAILABLE INFORMATION SOURCES:
1. OPPONENT STATS: Detailed playing patterns (VPIP, aggression, fold rates) - Critical for reading opponents
2. HAND ANALYSIS: Mathematical hand strength, probabilities, equity calculations - Essential for optimal play

You must output exactly this JSON format:
{{"need_opponent_stats": true, "need_hand_analysis": true, "ready_to_decide": false, "reasoning": ["why I need this info"]}}

DECISION FRAMEWORK:
- need_opponent_stats: Should be TRUE unless you have very recent reads on all opponents
- need_hand_analysis: Should be TRUE unless you have nuts/trash and decision is obvious  
- ready_to_decide: Only TRUE if you're confident without additional analysis
- reasoning: Explain your information needs (1-3 short reasons)

WHEN TO SKIP INFO GATHERING (rare cases):
- You have absolute nuts (straight flush, etc.) and want to bet/raise
- You have absolute trash and will fold regardless of opponent tendencies
- You have extensive recent data on opponents AND simple decision

EXAMPLES:
{{"need_opponent_stats": true, "need_hand_analysis": true, "ready_to_decide": false, "reasoning": ["need opponent tendencies", "hand strength unclear"]}}
{{"need_opponent_stats": true, "need_hand_analysis": false, "ready_to_decide": false, "reasoning": ["obvious strong hand but need opponent reads"]}}
{{"need_opponent_stats": false, "need_hand_analysis": false, "ready_to_decide": true, "reasoning": ["have nuts will bet max"]}}

Output JSON only:"""
        if self.log_prompts:
            self.logger.info('Prompt submitted:')
            self.logger.info(planning_prompt)
        raw_text = self.client.submit_prompt_return_response(planning_prompt)
        raw_text = (raw_text or '').strip()
        raw_text = raw_text.replace("```json", "").replace("```", "")
        planning = self._parse_planning_json(raw_text)
        
        if self.verbose:
            reasons = " | ".join(planning.get("reasoning", ["No reasoning provided"]))
            self.logger.info(f"{self.name} Planning: opponent_stats={planning.get('need_opponent_stats', False)}, "
                       f"hand_analysis={planning.get('need_hand_analysis', False)}, "
                       f"ready={planning.get('ready_to_decide', True)} | {reasons}")
        
        return planning

    def _parse_planning_json(self, raw_text: str) -> dict:
        try:
            return json.loads(raw_text)
        except:
            pass
        import re
        json_match = re.search(r'\{[^}]*\}', raw_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        try:
            cleaned = raw_text.strip()
            if not cleaned.startswith('{'):
                start = cleaned.find('{')
                if start != -1:
                    cleaned = cleaned[start:]
            if not cleaned.endswith('}'):
                end = cleaned.rfind('}')
                if end != -1:
                    cleaned = cleaned[:end+1]
            cleaned = re.sub(r'(\w+):', r'"\1":', cleaned)
            cleaned = re.sub(r':\s*([^",\[\]{}]+)(?=[,}])', r': "\1"', cleaned)
            return json.loads(cleaned)
        except:
            pass
        try:
            result = {
                "need_opponent_stats": False,
                "need_hand_analysis": False,
                "ready_to_decide": True,
                "reasoning": ["parsing fallback"]
            }
            if re.search(r'"?need_opponent_stats"?\s*:\s*true', raw_text, re.IGNORECASE):
                result["need_opponent_stats"] = True
            if re.search(r'"?need_hand_analysis"?\s*:\s*true', raw_text, re.IGNORECASE):
                result["need_hand_analysis"] = True
            if re.search(r'"?ready_to_decide"?\s*:\s*false', raw_text, re.IGNORECASE):
                result["ready_to_decide"] = False
            reasoning_match = re.search(r'"?reasoning"?\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
            if reasoning_match:
                reasons_text = reasoning_match.group(1)
                reasons = re.findall(r'"([^"]+)"', reasons_text)
                if reasons:
                    result["reasoning"] = reasons
            return result
        except:
            pass
        if self.verbose:
            self.logger.warning(f"All planning JSON parsing failed, raw output: {raw_text[:200]}...")
        return {
            "need_opponent_stats": False,
            "need_hand_analysis": False,
            "ready_to_decide": True,
            "reasoning": ["JSON parse failed, proceeding with basic info"]
        }

    def _gather_information(
        self,
        planning_decision: dict,
        player: Player,
        game_state: GameState,
        stats_tracker: Optional[StatsTracker]
    ) -> dict:
        gathered = {"basic_info": self._get_basic_prompt_info(player, game_state)}
        if planning_decision.get("need_opponent_stats", False) and stats_tracker:
            opponent_stats = stats_tracker.get_opponent_stats_summary(player.name)
            gathered["opponent_stats"] = opponent_stats
            if self.verbose:
                self.logger.info(f"{self.name}: Gathered opponent stats")
                #self.logger.info(f"{self.name}: Gathered opponent stats: {len(opponent_stats.split(chr(10)))} opponent(s) analyzed")
        if planning_decision.get("need_hand_analysis", False):
            try:
                feature_reporter = FeatureReporter(player.hole_cards, game_state)
                hand_features = feature_reporter.get_hand_features()
                gathered["hand_analysis"] = hand_features
                if self.verbose:
                    self.logger.info(f"{self.name}: Gathered hand analysis: {len(hand_features)} feature(s)")
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"{self.name}: Hand analysis failed: {e}")
                gathered["hand_analysis"] = ["Hand analysis unavailable"]
        return gathered

    def _get_basic_prompt_info(self, player: Player, game_state: GameState) -> str:
        info = player.get_hand_info(game_state)
        valid_actions = player.get_valid_actions(game_state)
        lines = []
        lines.append(f"You are {player.name}, an AI poker agent competing in a Texas Hold'em hand.")
        lines.append("All opponents at the table are also AI agents who receive similar information.")
        lines.append("Your objective is to maximize your total chip winnings over many hands, not just in a single hand.")
        lines.append(f"Your hole cards: {info['my_hole_cards'][0]} {info['my_hole_cards'][1]}")
        lines.append(f"Your chips: {info['my_chips']}")
        lines.append(f"Current pot: {info['pot']}")
        if info['board']:
            board_str = " ".join(str(card) for card in info['board'])
            lines.append(f"Community cards: {board_str}")
        else:
            lines.append("Community cards: None dealt yet")
        lines.append(f"Betting round: {info['betting_round_name']}")
        
        
        
        if ActionType.CHECK in valid_actions:
            lines.append("No bet to call - you can check or bet")
        elif info['amount_to_call'] > 0:
            lines.append(f"You need to call {info['amount_to_call']} to stay in the hand")
        else:
            lines.append("All opponents have folded or acted - you can bet or fold")

            
            
        
        lines.append(f"Your current bet this round: {info['my_current_bet']}")
        lines.append(f"\nOpponents remaining: {info['opponents_remaining']}")
        for player_info in info['active_players']:
            if player_info['name'] != player.name:
                status = " (ALL-IN)" if player_info['is_all_in'] else ""
                lines.append(f"  {player_info['name']}: {player_info['chips']} chips, bet {player_info['current_bet']} this round{status}")
        lines.append(f"\nActions this betting round:")
        if info['actions_this_round']:
            for action in info['actions_this_round']:
                amt = action.get('amount') or 0
                if amt > 0:
                    lines.append(f"  {action['player']}: {action['action']} {amt}")
                else:
                    lines.append(f"  {action['player']}: {action['action']}")
        else:
            lines.append("  No actions yet this round")
        previous_rounds = [a for a in info['full_hand_history'] if a['round'] != info['betting_round_name']]
        if previous_rounds:
            lines.append(f"\nPrevious betting rounds:")
            current_round = None
            for action in previous_rounds:
                if action['round'] != current_round:
                    current_round = action['round']
                    lines.append(f"  {current_round}:")
                amt = action.get('amount') or 0
                if amt > 0:
                    lines.append(f"    {action['player']}: {action['action']} {amt}")
                else:
                    lines.append(f"    {action['player']}: {action['action']}")
        return "\n".join(lines)



    def _repair_decision_json(self, raw_text: str, legal_str: str, max_bet: int, max_raise: int) -> dict:
        """Simple repair with logging and emergency fallback"""
        if self.verbose:
            self.logger.warning(f"{self.name}: Initial JSON parsing failed. Raw response: '{raw_text[:500]}...'")
        
        # Try the LLM repair
        try:
            repair_prompt = f"""
            You will be given model output that SHOULD be a single JSON object with EXACTLY these keys:
            {{
              "action": "fold" | "check" | "call" | "bet" | "raise" | "all_in",
              "amount": null | <integer>,
              "reasons": [ "<brief bullet>", ... ]   // 1-5 strings, ≤12 words each
            }}
            
            Legal actions: [{legal_str}]
            Constraints:
            - For "bet": 1 ≤ amount ≤ {max_bet}
            - For "raise": 1 ≤ amount ≤ {max_raise}
            - For "check", "call", "fold", "all_in": amount MUST be null.
            
            Your task: FIX the input so it becomes a VALID JSON object that obeys the schema and constraints.
            Return ONLY the fixed JSON object. No extra text.
            
            INPUT:
            <<<
            {raw_text}
            >>>
            """
            if self.log_prompts:
                self.logger.info('Json repair prompt submitted:')
                self.logger.info(repair_prompt)
            fixed = self.client.submit_prompt_return_response(repair_prompt)
            fixed = (fixed or '').strip()
            
            if not fixed:
                raise ValueError("Empty response from repair prompt")
                
            if self.verbose:
                self.logger.info(f"{self.name}: Repair attempt returned: '{fixed[:200]}...'")
            
            return json.loads(fixed)
        
        except Exception as repair_error:
            if self.verbose:
                self.logger.error(f"{self.name}: Repair also failed: {repair_error}")
            
            # Ultimate fallback - return a safe default action
            legal_actions = [ActionType(a.strip().lower()) for a in legal_str.split(',')]
            if ActionType.FOLD in legal_actions:
                default_action = "fold"
            elif ActionType.CHECK in legal_actions:
                default_action = "check"
            elif ActionType.CALL in legal_actions:
                default_action = "call"
            else:
                default_action = legal_actions[0].value
                
            if self.verbose:
                self.logger.warning(f"{self.name}: Using emergency fallback action: {default_action}")
                
            return {
                "action": default_action,
                "amount": None,
                "reasons": ["Emergency fallback due to parsing failure"]
            }
        
        
    def _maybe_reconsider_bluff(
        self,
        combined_info: str,
        decision: dict,
        allowed_actions: list,
        call_needed: int,
        max_bet: int,
        max_raise: int,
        player: Player
    ) -> dict | None:
        legal_str = ', '.join(a.value for a in allowed_actions)
        constraint_lines = []
        for a in allowed_actions:
            if a == ActionType.BET:
                constraint_lines.append(f'- If action is "bet": amount must be 1-{max_bet}')
            elif a == ActionType.RAISE:
                constraint_lines.append(f'- If action is "raise": amount must be 1-{max_raise}')
            elif a in [ActionType.CHECK, ActionType.CALL, ActionType.FOLD, ActionType.ALL_IN]:
                constraint_lines.append(f'- If action is "{a.value}": amount must be null')
        constraints_text = '\n    '.join(constraint_lines)
    
        prompt = f"""You are reconsidering your poker decision. 
        Your response must be a single JSON object with no additional text, explanations, or formatting.
        
        SITUATION:
            {combined_info}
        
        ORIGINAL DECISION:
            action="{decision.get('action')}", amount={decision.get('amount')}
        
        LEGAL ACTIONS:
            {legal_str}
        
        CONSTRAINTS:
            {constraints_text}
        
        Your response must start with {{ and end with }} and contain only this JSON structure:
        {{
          "switch_to_bluff": true|false,
          "action": "bet"|"raise"|null,
          "amount": <int>|null,
          "reasons": ["reason1", "reason2"]
        }}
        
        Rules:
        - "switch_to_bluff": true if you want to change to a bluff/semi-bluff, otherwise false
        - "action": "bet" or "raise" if switching, null otherwise
        - "amount": positive integer within legal limits if switching, null otherwise
        - "reasons": 1–2 short reasons (<=80 chars each)
        
        Do not include any other text. Do not use markdown formatting. Do not use code blocks. 
        Do not add explanations before or after the JSON. 
        Your entire response must be exactly the JSON object and nothing else.
        
        Start your response with the opening brace:
        """


        if self.log_prompts:
            self.logger.info(f'{self.name}: Bluff reconsideration prompt submitted:')
            self.logger.info(prompt)
    
        raw = self.client.submit_prompt_return_response(prompt)
        raw = (raw or '').strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        
    
        try:
            parsed = json.loads(raw)
        except Exception as e:
            self.logger.warning(
                f'{self.name}: Bluff reconsideration JSON parse failed: {e}; raw={raw[:400]}'
            )
            # Try repairing like we do in normal decision flow
            repaired = self._repair_decision_json(
                raw,
                legal_str,
                max_bet,
                max_raise
            )
            if repaired:
                self.logger.info(f'{self.name}: Repaired bluff reconsideration JSON: {repaired}')
                parsed = repaired
            else:
                self.logger.info(f'{self.name}: Repaired bluff reconsideration JSON failed')
                return None
    
        if not isinstance(parsed, dict):
            self.logger.warning(f'{self.name}: Bluff reconsideration returned non-object; raw={raw[:400]}')
            return None
    
        if not parsed.get('switch_to_bluff'):
            self.logger.info(f'{self.name}: Choosing not to bluff')
            return None
    
        action = parsed.get('action')
        amount = parsed.get('amount')
        reasons = parsed.get('reasons')
    
        if not isinstance(action, str):
            self.logger.warning(f'{self.name}: Bluff reconsideration missing/invalid action; obj={parsed}')
            return None
    
        action_lc = action.lower()
        atype = next((a for a in allowed_actions if a.value == action_lc), None)
        if atype is None:
            self.logger.warning(f'{self.name}: Bluff reconsideration illegal action "{action}"; allowed={legal_str}')
            return None
    
        if atype == ActionType.BET and not isinstance(amount, int):
            self.logger.warning(f'{self.name}: Bluff reconsideration bet missing/invalid amount; obj={parsed}')
            return None
        if atype == ActionType.RAISE and not isinstance(amount, int):
            self.logger.warning(f'{self.name}: Bluff reconsideration raise missing/invalid amount; obj={parsed}')
            return None
    
        if atype == ActionType.BET and not (1 <= amount <= max_bet):
            self.logger.warning(f'{self.name}: Bluff reconsideration bet amount out of range 1-{max_bet}; got={amount}')
            return None
        if atype == ActionType.RAISE and not (1 <= amount <= max_raise):
            self.logger.warning(f'{self.name}: Bluff reconsideration raise amount out of range 1-{max_raise}; got={amount}')
            return None
    
        if isinstance(reasons, str):
            reasons = [reasons]
        if not isinstance(reasons, list):
            reasons = []
    
        return {'action': action_lc, 'amount': amount, 'reasons': reasons}


    

    def _infer_phase_from_board(self, game_state: GameState) -> str:
        board = getattr(game_state, 'board', None) or []
        n = len(board)
        if n == 0:
            return 'preflop'
        if n == 3:
            return 'flop'
        if n == 4:
            return 'turn'
        if n == 5:
            return 'river'
        return 'complete'


    def _make_betting_decision(
        self,
        player: Player,
        game_state: GameState,
        valid_actions: List[ActionType],
        gathered_info: dict
    ) -> Action:
        """Two-stage decision with all original robustness: 1) Action type, 2) Sizing if bet/raise"""
        
        # Preserve original action filtering logic
        call_needed = max(0, game_state.current_bet - player.current_bet)
        max_bet = player.chips
        max_raise = max(0, player.chips - call_needed)
        allowed_actions = []
        for a in valid_actions:
            if a == ActionType.BET and max_bet < 1:
                continue
            if a == ActionType.RAISE and max_raise < 1:
                continue
            allowed_actions.append(a)
        
        legal_str = ", ".join(a.value for a in allowed_actions)
        
        # Build combined info (preserve original logic)
        info_sections = [gathered_info["basic_info"]]
        if "opponent_stats" in gathered_info:
            info_sections.append(f"\nOPPONENT STATISTICS:\n{gathered_info['opponent_stats']}")
        if "hand_analysis" in gathered_info:
            analysis_str = "\n".join(gathered_info["hand_analysis"])
            info_sections.append(f"\nHAND ANALYSIS:\n{analysis_str}")
        combined_info = "\n".join(info_sections)
        
        # STAGE 1: Get action type decision (no amounts)
        action_type_decision = self._get_action_type_decision(
            player, game_state, allowed_actions, combined_info
        )
        
        action_type = action_type_decision["action_type"]
        base_reasons = action_type_decision.get("reasons", [])
        
        # Preserve original bluff reconsideration logic
        bluffable = action_type in [ActionType.CHECK, ActionType.CALL]
        has_aggressive = any(a in (ActionType.BET, ActionType.RAISE) for a in allowed_actions)
        if bluffable and has_aggressive:
            maybe_bluff = self._maybe_reconsider_bluff(
                combined_info=combined_info,
                decision={"action": action_type.value, "amount": None},
                allowed_actions=allowed_actions,
                call_needed=call_needed,
                max_bet=max_bet,
                max_raise=max_raise,
                player=player
            )
            if maybe_bluff:
                # Override with bluff decision
                action_type = ActionType(maybe_bluff["action"])
                base_reasons = maybe_bluff.get("reasons", base_reasons)
                if self.verbose:
                    self.logger.info(f'{self.name}: Converted to bluff: {action_type.value}')
        
        # STAGE 2: Get sizing if betting/raising
        if action_type in [ActionType.BET, ActionType.RAISE]:
            amount = self._get_sizing_decision(
                player, game_state, action_type, combined_info, base_reasons
            )
            final_decision = {"action": action_type.value, "amount": amount, "reasons": base_reasons}
        else:
            final_decision = {"action": action_type.value, "amount": None, "reasons": base_reasons}
        
        # Preserve original validation logic
        return self._validate_and_create_action(
            final_decision, allowed_actions, call_needed, max_bet, max_raise, player
        )
    
    
    def _get_action_type_decision(
        self,
        player: Player,
        game_state: GameState,
        allowed_actions: List[ActionType],
        combined_info: str
    ) -> dict:
        """Get action type only (fold/check/call/bet/raise/all_in) without amounts"""
        
        legal_str = ", ".join(a.value for a in allowed_actions)
        
        # Preserve original phase logic
        phase = self._infer_phase_from_board(game_state)
        if phase == 'preflop' and getattr(player, 'is_small_blind', False):
            sb_hint = 'As SB preflop, avoid folding marginal hands; prefer limp/call/raise over fold unless dominated.'
        else:
            sb_hint = ''
            
        # Blind hint
        blind_defense_hint = ""
        if phase == 'preflop' and player.current_bet > 0:
            call_needed = max(0, game_state.current_bet - player.current_bet)
            if call_needed > 0:
                pot_odds = game_state.pot / call_needed
                blind_defense_hint = f"""
        CRITICAL BLIND DEFENSE GUIDANCE:
        - You have blinds posted and face a small raise - you are getting excellent pot odds
        - Small blind defense: With 3:1+ pot odds, fold only the weakest 15-20% of hands (like 72o, 83o, 92o)
        - Big blind defense: With 2:1+ pot odds, defend with 60-70% of hands
        - Your blind investment is already spent - focus on whether the call is profitable going forward
        - Current pot odds: {pot_odds:.1f}:1"""

        
        
        action_type_prompt = f"""{self.player_profile}

        You are deciding your poker action type. Bet/raise sizing will be handled in a separate step. Your response must be a single JSON object with no additional text, explanations, or formatting.
        
        SITUATION:
        {combined_info}
        
        LEGAL ACTIONS: [{legal_str}]
        
        DECISION FRAMEWORK:
        - Choose the action that maximizes long-term chip EV
        - Focus on the strategic decision (fold vs stay vs aggress)
        - Don't worry about bet/raise amounts - that comes next
        {sb_hint}
        {blind_defense_hint}
        
        Your response must start with {{ and end with }} and contain only this JSON structure:
        {{
          "action_type": "ACTION_HERE",
          "reasons": ["reason1", "reason2"]
        }}
        
        Do not include any other text. Do not use markdown formatting. Do not use code blocks. Do not add explanations before or after the JSON. Your entire response must be exactly the JSON object and nothing else.
        
        Start your response with the opening brace:"""
        
        if self.log_prompts:
            self.logger.info(f'{self.name}: Action type prompt submitted')
            self.logger.info(action_type_prompt)
        
        raw_text = self.client.submit_prompt_return_response(action_type_prompt)
        raw_text = (raw_text or '').strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        # Use similar parsing strategy as original with repair fallback
        try:
            decision = json.loads(raw_text)
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"{self.name}: Action type JSON parsing failed: {e}")
            # Use repair logic similar to original
            decision = self._repair_action_type_json(raw_text, legal_str, allowed_actions)
        
        # Validate action type
        action_str = decision.get("action_type", "").lower()
        try:
            action_type = ActionType(action_str)
            if action_type not in allowed_actions:
                raise ValueError(f"Illegal action: {action_str}")
        except ValueError:
            if self.verbose:
                self.logger.warning(f"{self.name}: Invalid action type '{action_str}', using fallback")
            # Fallback to safe action (preserve original logic)
            if ActionType.FOLD in allowed_actions:
                action_type = ActionType.FOLD
            elif ActionType.CHECK in allowed_actions:
                action_type = ActionType.CHECK
            else:
                action_type = allowed_actions[0]
        
        return {
            "action_type": action_type,
            "reasons": decision.get("reasons", ["Action type decision"])
        }
    
    
    def _repair_action_type_json(self, raw_text: str, legal_str: str, allowed_actions: List[ActionType]) -> dict:
        """Repair malformed action type JSON (similar to original repair logic)"""
        if self.verbose:
            self.logger.warning(f"{self.name}: Attempting action type JSON repair")
        
        try:
            # Try LLM repair similar to original
            repair_prompt = f"""Fix this JSON to have exactly these keys:
    {{
      "action_type": "fold|check|call|bet|raise|all_in",
      "reasons": ["reason1", "reason2"]
    }}
    
    Legal actions: [{legal_str}]
    Return ONLY the fixed JSON.
    
    INPUT: {raw_text}"""
            
            if self.log_prompts:
                self.logger.info('Action type repair prompt submitted')
            
            fixed = self.client.submit_prompt_return_response(repair_prompt)
            fixed = (fixed or '').strip()
            return json.loads(fixed)
        
        except Exception:
            if self.verbose:
                self.logger.error(f"{self.name}: Action type repair failed, using emergency fallback")
            
            # Emergency fallback (preserve original logic)
            if ActionType.FOLD in allowed_actions:
                fallback_action = "fold"
            elif ActionType.CHECK in allowed_actions:
                fallback_action = "check"
            else:
                fallback_action = allowed_actions[0].value
                
            return {
                "action_type": fallback_action,
                "reasons": ["Emergency fallback due to parsing failure"]
            }
    
    
    def _get_sizing_decision(
        self,
        player: Player,
        game_state: GameState,
        action_type: ActionType,
        combined_info: str,
        base_reasons: List[str]
    ) -> int:
        """Get specific bet/raise sizing using BetSizingAnalyzer"""
        
        try:
            # Generate sizing analysis using the new analyzer
            sizing_analyzer = BetSizingAnalyzer(player, game_state, action_type, self.table_config)
            sizing_context = sizing_analyzer.get_all_sizing_context()
            max_affordable = sizing_analyzer.max_affordable
            
            # Build sizing-focused prompt
            context_str = "\n".join(sizing_context)
            action_desc = "bet" if action_type == ActionType.BET else "raise by"
            
            # Include relevant strategic context from earlier analysis
            strategic_summary = []
            if "hand_analysis" in combined_info:
                # Extract key points for sizing context
                strategic_summary.append("Hand analysis considered in action decision")
            if "opponent_stats" in combined_info:
                strategic_summary.append("Opponent tendencies considered in action decision") 
            
            strategic_context = "; ".join(strategic_summary) if strategic_summary else "Standard sizing principles apply"
            
            sizing_prompt = f"""{self.player_profile}

            You decided to {action_desc}. Now choose the optimal size. Your response must be a single JSON object with no additional text, explanations, or formatting.
            
            SIZING ANALYSIS:
            {context_str}
            
            STRATEGIC CONTEXT: {strategic_context}
            ACTION REASONING: {"; ".join(base_reasons)}
            
            SIZING PRINCIPLES:
            - Value bets: Size to get called by worse hands
            - Bluffs: Size to fold out better hands efficiently  
            - Protection: Size to deny proper odds to draws
            - Stack management: Consider future betting rounds
            
            Your response must start with {{ and end with }} and contain only this JSON structure:
            {{
              "amount": <integer_1_to_{max_affordable}>,
              "reasoning": ["sizing reason 1", "sizing reason 2"]
            }}
            
            Do not include any other text. Do not use markdown formatting. Do not use code blocks. Do not add explanations before or after the JSON. Your entire response must be exactly the JSON object and nothing else.
            
            Start your response with the opening brace:"""
            
            if self.log_prompts:
                self.logger.info(f'{self.name}: Sizing prompt submitted')
                self.logger.info(sizing_prompt)
                
            raw_response = self.client.submit_prompt_return_response(sizing_prompt)
            raw_response = (raw_response or '').strip()
            raw_response = raw_response.replace("```json", "").replace("```", "").strip()
            
            # Parse sizing response with error handling
            try:
                sizing_decision = json.loads(raw_response)
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"{self.name}: Sizing JSON parsing failed: {e}")
                sizing_decision = self._repair_sizing_json(raw_response, max_affordable)
            
            amount = sizing_decision.get("amount")
            reasoning = sizing_decision.get("reasoning", [])
            
            # Validate amount
            if not isinstance(amount, int) or amount < 1 or amount > max_affordable:
                if self.verbose:
                    self.logger.warning(f"{self.name}: Invalid sizing amount {amount}, using fallback")
                raise ValueError(f"Invalid amount: {amount}")
                
            if self.verbose:
                reason_text = "; ".join(reasoning) if reasoning else "no sizing reasoning"
                self.logger.info(f"{self.name}: Sizing decision: {action_desc} {amount} because {reason_text}")
                
            return amount
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"{self.name}: Sizing decision failed: {e}, using fallback")
            
            # Fallback to reasonable defaults (preserve original-style logic)
            call_needed = max(0, game_state.current_bet - player.current_bet)
            max_affordable = player.chips if action_type == ActionType.BET else player.chips - call_needed
            
            if action_type == ActionType.BET:
                # Default to half pot bet
                fallback = min(max_affordable, max(1, game_state.pot // 2))
            else:  # RAISE
                # Default to third pot raise  
                fallback = min(max_affordable, max(1, game_state.pot // 3))
                
            if self.verbose:
                self.logger.info(f"{self.name}: Using fallback sizing: {fallback}")
                
            return fallback
    
    
    def _repair_sizing_json(self, raw_text: str, max_affordable: int) -> dict:
        """Repair malformed sizing JSON"""
        if self.verbose:
            self.logger.warning(f"{self.name}: Attempting sizing JSON repair")
        
        try:
            repair_prompt = f"""Fix this JSON to have exactly these keys:
            {{
              "amount": <integer_1_to_{max_affordable}>,
              "reasoning": ["reason1", "reason2"]
            }}
            
            Return ONLY the fixed JSON.
            
            INPUT: {raw_text}"""
            
            if self.log_prompts:
                self.logger.info('Sizing repair prompt submitted')
            
            fixed = self.client.submit_prompt_return_response(repair_prompt)
            fixed = (fixed or '').strip()
            return json.loads(fixed)
        
        except Exception:
            if self.verbose:
                self.logger.error(f"{self.name}: Sizing repair failed, using emergency fallback")
            
            # Emergency fallback
            fallback_amount = min(max_affordable, max(1, max_affordable // 3))
            return {
                "amount": fallback_amount,
                "reasoning": ["Emergency fallback sizing"]
            }
        
        
   
    
    def _validate_and_create_action(
        self,
        decision: dict,
        allowed_actions: List[ActionType],
        call_needed: int,
        max_bet: int,
        max_raise: int,
        player: Player
    ) -> Action:
        if not isinstance(decision, dict):
            raise TypeError('LLM JSON must be an object')
        action_str = decision.get('action')
        if not isinstance(action_str, str):
            raise ValueError('Missing or invalid "action"')
        try:
            atype = ActionType(action_str.lower())
        except ValueError as e:
            raise ValueError(f'Unknown action "{action_str}"') from e
        if atype not in allowed_actions:
            raise ValueError(f'Illegal action for state: {atype.value} not in {[a.value for a in allowed_actions]}')
        amount_field = decision.get('amount', None)
        amt = 0
        if atype in (ActionType.BET, ActionType.RAISE):
            if amount_field is None:
                raise ValueError('amount must be provided for bet/raise')
            if not isinstance(amount_field, int):
                raise TypeError('amount must be an integer for bet/raise')
            if amount_field < 1:
                raise ValueError('amount must be >= 1 for bet/raise')
            max_afford = max_bet if atype == ActionType.BET else max_raise
            if max_afford <= 0:
                raise ValueError('Insufficient chips for requested bet/raise')
            if amount_field > max_afford:
                raise ValueError(f'amount {amount_field} exceeds max affordable {max_afford}')
            amt = amount_field
        elif atype == ActionType.CALL:
            amt = call_needed
        else:
            if amount_field not in (None, 0):
                raise ValueError(f'Non-bet/raise action must not include amount, got {amount_field}')
        reasons = decision.get('reasons', None)
        if reasons is not None:
            if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
                raise TypeError('"reasons" must be a list of strings')
            reasons = [r.strip() for r in reasons if r and isinstance(r, str)]
        if self.verbose:
            if atype == ActionType.BET:
                amount_phrase = f" bet {amt}"
            elif atype == ActionType.RAISE:
                amount_phrase = f" raise {amt}"
            elif atype == ActionType.ALL_IN:
                amount_phrase = f" all-in {player.chips}"
            else:
                amount_phrase = ""
            reason_text = "; ".join(reasons) if reasons else "no reasons provided"
            self.logger.info(f"{self.name}: Final decision: {atype.value}{amount_phrase} because {reason_text}")
        return Action(atype, amt, reasons=reasons)





class RandomAgent(Agent):
    #def get_action(self, player: Player, game_state: GameState, valid_actions: List[ActionType]) -> Action:
    def get_action(
            self,
            player: Player,
            game_state: GameState,
            valid_actions: List[ActionType],
            stats_tracker: Optional[StatsTracker] = None
        ) -> Action:
        action_type = random.choice(valid_actions)
        
        if action_type == ActionType.BET:
            amount = random.randint(1, min(player.chips, 10))
            return Action(ActionType.BET, amount)
        elif action_type == ActionType.RAISE:
            call_amount = game_state.current_bet - player.current_bet
            max_raise = player.chips - call_amount
            if max_raise > 0:
                amount = random.randint(1, min(max_raise, 10))
                return Action(ActionType.RAISE, amount)
            else:
                return Action(ActionType.ALL_IN)
        else:
            return Action(action_type)



    
