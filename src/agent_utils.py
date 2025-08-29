# -*- coding: utf-8 -*-
"""
agent_utils.py

Utilities that agents call
"""
import os
from dotenv import load_dotenv
import numpy as np
import requests
import time
from typing import List, TYPE_CHECKING
import yaml

from src.core_poker_mechanics import Card
from src.poker_types import ActionType
import src.feature_engineering as fe
from src.logging_config import logger
if TYPE_CHECKING:
    from src.game import GameState, Player, TableConfig

load_dotenv()

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}

class BetSizingAnalyzer:
    """
    Analyze betting context and generate sizing recommendations.
    Similar to FeatureReporter - does the calculations, returns descriptive strings.
    """
    
    def __init__(
            self,
            player: 'Player',
            game_state: 'GameState',
            action_type: ActionType,
            table_config: 'TableConfig'
        ):
        self.player = player
        self.game_state = game_state
        self.action_type = action_type  # BET or RAISE
        self.table_config = table_config
        
        # Basic financial state
        self.pot = game_state.pot
        self.my_chips = player.chips
        self.current_bet = game_state.current_bet
        self.my_current_bet = player.current_bet
        self.call_amount = max(0, self.current_bet - self.my_current_bet)
        
        # Calculate max affordable
        if action_type == ActionType.BET:
            self.max_affordable = self.my_chips
        else:  # RAISE
            self.max_affordable = self.my_chips - self.call_amount
    
    def get_financial_context_string(self) -> str:
        """Basic financial situation"""
        if self.action_type == ActionType.BET:
            return f"Pot: {self.pot}, Your chips: {self.my_chips}, Can bet: 1-{self.max_affordable}"
        else:
            total_commitment = self.call_amount + 1
            return f"Pot: {self.pot}, Your chips: {self.my_chips}, Call: {self.call_amount}, Can raise by: 1-{self.max_affordable} (total commitment: {total_commitment}-{self.my_chips})"
    
    def get_blind_context_string(self) -> str:
        """Reference blind sizes for perspective"""
        sb = self.table_config.small_blind
        bb = self.table_config.big_blind
        return f"Small blind: {sb}, Big blind: {bb} (for sizing reference)"
    
    def get_pot_based_sizes_string(self) -> str:
        """Standard pot-based sizing options"""
        pot_fractions = [0.25, 0.33, 0.5, 0.67, 1.0, 1.5]
        sizes = []
        
        for fraction in pot_fractions:
            amount = max(1, int(self.pot * fraction))
            if amount <= self.max_affordable:
                if fraction == 0.25:
                    desc = "1/4 pot"
                elif fraction == 0.33:
                    desc = "1/3 pot"
                elif fraction == 0.5:
                    desc = "1/2 pot"
                elif fraction == 0.67:
                    desc = "2/3 pot"
                elif fraction == 1.0:
                    desc = "pot"
                elif fraction == 1.5:
                    desc = "1.5x pot"
                
                sizes.append(f"{desc}: {amount}")
        
        if sizes:
            return "Pot-based sizes: " + ", ".join(sizes)
        else:
            return f"All standard pot sizes exceed your maximum ({self.max_affordable})"
    
    def get_blind_based_sizes_string(self) -> str:
        """Standard big blind multiple sizes"""
        bb = self.table_config.big_blind
        multiples = [2, 3, 5, 8, 12, 20]
        sizes = []
        
        for mult in multiples:
            amount = bb * mult
            if amount <= self.max_affordable:
                sizes.append(f"{mult}x BB: {amount}")
        
        if sizes:
            return "Big blind multiples: " + ", ".join(sizes)
        else:
            return f"Standard BB multiples exceed your maximum ({self.max_affordable})"
    
    def get_stack_depth_context_string(self) -> str:
        """Stack depth relative to pot and blinds"""
        bb = self.table_config.big_blind
        stack_in_bb = self.my_chips / bb if bb > 0 else 0
        pot_to_stack_ratio = self.pot / self.my_chips if self.my_chips > 0 else 0
        
        if stack_in_bb > 100:
            depth = "deep"
        elif stack_in_bb > 40:
            depth = "medium"
        elif stack_in_bb > 20:
            depth = "short"
        else:
            depth = "very short"
        
        return f"Stack depth: {depth} ({stack_in_bb:.1f} BB), Pot is {pot_to_stack_ratio:.1%} of your stack"
    
    def get_position_context_string(self) -> str:
        """Position-based sizing considerations"""
        active_players = len(self.game_state.get_active_players())
        dealer_pos = self.game_state.dealer_position
        player_index = self.game_state.players.index(self.player)
        position = (player_index - dealer_pos) % len(self.game_state.players)
        
        if active_players == 2:
            pos_desc = "heads-up"
        elif position == 0:
            pos_desc = "in position (dealer)"
        elif position == len(self.game_state.players) - 1:
            pos_desc = "out of position (last to act pre-flop)"
        else:
            pos_desc = f"position {position}"
        
        return f"Position: {pos_desc} vs {active_players-1} opponent(s)"
    
    def get_betting_round_context_string(self) -> str:
        """Context about which betting round for sizing"""
        round_names = ["preflop", "flop", "turn", "river"]
        round_name = round_names[self.game_state.betting_round]
        
        # Count actions this round for aggression level
        recent_actions = self.game_state.get_actions_since_board_change()
        aggressive_actions = sum(1 for a in recent_actions 
                               if a.action.action_type in [ActionType.BET, ActionType.RAISE])
        
        if aggressive_actions == 0:
            aggression = "no prior aggression"
        elif aggressive_actions == 1:
            aggression = "some aggression"
        else:
            aggression = "heavy aggression"
        
        return f"Betting round: {round_name}, {aggression} this round"
    
    def get_all_sizing_context(self) -> List[str]:
        """Get all sizing analysis as list of descriptive strings (like FeatureReporter)"""
        context_strings = [
            self.get_financial_context_string(),
            self.get_blind_context_string(),
            self.get_pot_based_sizes_string(),
            self.get_blind_based_sizes_string(),
            self.get_stack_depth_context_string(),
            self.get_position_context_string(),
            self.get_betting_round_context_string()
        ]
        
        return context_strings





def card_to_str(card: Card) -> str:
    """Convert a Card object into 'Ad', 'Ks', etc. for encode_game_state_from_strings."""
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
    return rank_map[card.rank] + suit_map[card.suit.value]



class FeatureReporter:
    """
    Generate descriptive feature strings from a card dict like:
    {'my_hole_cards': ['8d', '4c'], 'board': ['Ac', '4s', '9d']}
    """

    def __init__(self, hole_cards, game_state: 'GameState'):
        self.hole_cards = hole_cards
        self.game_state = game_state
        self.card_dict = self.get_cards()

        self.hole = [self.parse_card(c) for c in self.card_dict['my_hole_cards']]
        self.board = [self.parse_card(c) for c in self.card_dict['board']]

        self.hole_ranks = np.array([r for r, s in self.hole], dtype=np.int32)
        self.hole_suits = np.array([s for r, s in self.hole], dtype=np.int32)
        self.board_ranks = np.array([r for r, s in self.board], dtype=np.int32)
        self.board_suits = np.array([s for r, s in self.board], dtype=np.int32)

        self.num_opponents = len(self.game_state.get_active_players()) - 1
        
        
    def get_cards(self) -> dict:
        """Get dictionary with player hole cards and board cards"""
        # old:
        # return {
        #     'my_hole_cards': [card_to_str(c) for c in self.hole_cards],
        #     'board': [card_to_str(c) for c in self.game_state.board.copy()]
        # }
    
        hole = self.hole_cards or []
        board = (self.game_state.board or []).copy()
        return {
            'my_hole_cards': [card_to_str(c) for c in hole],
            'board': [card_to_str(c) for c in board]
        }

    def parse_card(self, card_str: str) -> tuple[int, int]:
        """Convert string like 'Ad' to (rank, suit)."""
        rank_str_to_int = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
            '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
            'K': 13, 'A': 14
        }
        suit_str_to_int = {'c': 0, 'h': 1, 'd': 2, 's': 3}
        rank = rank_str_to_int[card_str[0].upper()]
        suit = suit_str_to_int[card_str[1].lower()]
        return rank, suit
    

    def hand_strength_string(self) -> str:
        """Return a descriptive string with normalized hand strength."""
        
        scalar = fe.extract_hand_strength_scalar(
            hole_ranks = self.hole_ranks,
            hole_suits = self.hole_suits,
            board_ranks = self.board_ranks,
            board_suits = self.board_suits   
        )
        return f"Approximate hand strength {scalar:.1f} (0 to 1 scale)"
    
    
    def current_hand_string(self, raw = False) -> str:
        hand_name = fe.best_hand_string(
            hole_ranks = self.hole_ranks,
            hole_suits = self.hole_suits,
            board_ranks = self.board_ranks,
            board_suits = self.board_suits   
        )
        if raw:
            return hand_name
        return f"Your current hand is {hand_name}"
    
    
    def suited_hole_string(self) -> str:
        """Return a descriptive string indicating whether hole cards are suited or not"""
        if self.hole_suits[0] == self.hole_suits[1]:
            return "Your pocket cards are suited"
        else:
            return "Your pocket cards are not suited"
        
    @staticmethod
    def fmt(p: float) -> str:
        pct = int(round(float(p) * 100))
        return f'{pct}%'
        
    def straight_probability_string(self) -> str:
        """Return a descriptive string indicating an approximate probability of getting a straight vs. opponents probability"""
        straight_prob = fe.calculate_straight_probability(
            hole_ranks=self.hole_ranks,
            hole_suits=self.hole_suits,
            board_ranks=self.board_ranks,
            board_suits=self.board_suits
        )

        opponent_straight_prob = fe.calculate_opponent_straight_probability(
            hole_ranks=self.hole_ranks,
            hole_suits=self.hole_suits,
            board_ranks=self.board_ranks,
            board_suits=self.board_suits,
            num_opponents=self.num_opponents
        )
        return f'Straight chance ~{self.fmt(straight_prob)}; opponents ~{self.fmt(opponent_straight_prob)}'
    
    
    def flush_probability_string(self) -> str:
        """Return a descriptive string indicating an approximate probability of getting a straight vs. opponents probability"""
        flush_prob = fe.calculate_flush_probability(
            hole_ranks=self.hole_ranks,
            hole_suits=self.hole_suits,
            board_ranks=self.board_ranks,
            board_suits=self.board_suits
        )

        opponent_flush_prob = fe.calculate_opponent_flush_probability(
            hole_ranks=self.hole_ranks,
            hole_suits=self.hole_suits,
            board_ranks=self.board_ranks,
            board_suits=self.board_suits,
            num_opponents=self.num_opponents
        )
        return f'Flush chance ~{self.fmt(flush_prob)}; opponents ~{self.fmt(opponent_flush_prob)}'
        
    
        
    def get_hand_features(self) -> List[str]:
        """Get descriptive strings about player hand based on feature_engineering.py logic"""
        feature_str_list = []
        
        board_hand = fe.best_board_pattern(board_ranks = self.board_ranks, board_suits = self.board_suits)
        
        if len(self.card_dict.get('board')) > 0:
            # Hand strength scalar string
            feature_str_list.append(self.hand_strength_string())

            # Name of hand
            feature_str_list.append(self.current_hand_string())
            
            # Odds of straight vs. opponent odds of straight
            feature_str_list.append(self.straight_probability_string())
            
            feature_str_list.append(self.flush_probability_string())
            
            # Board-only
            if board_hand:
                board_hand_str = f'Note: The board cards alone make a {board_hand}, so all players have at least the board {board_hand}'
                feature_str_list.append(board_hand_str)
            
        if len(self.card_dict.get('board')) == 0:
            # Pocket cards suited
            feature_str_list.append(self.suited_hole_string())

        return feature_str_list





class OpenRouterCompletionEngine:
    """
    
    Example Usage
    -------------
    engine = OpenRouterCompletionEngine()
    answer = engine.submit_prompt_return_response('The first person on the moon was')
    print(answer)
    
    """
    def __init__(
        self,
        model: str = 'openai/gpt-4o',
        token: str = os.getenv("OPEN_ROUTER_TOKEN"),
        url: str = 'https://openrouter.ai/api/v1/chat/completions'
        ):
        self.model = model
        self.token = token
        self.url = url
        game_cfg = config.get('game', {}) if isinstance(config, dict) else {}
        self.delay = float(game_cfg.get('llm_call_delay', 0.0))

    def submit_prompt(self, prompt: str, max_tokens: int | None = 40000, temperature: float = 0.7):
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
        }
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
    
        try:
            response = requests.post(
                self.url,
                headers={'Authorization': f'Bearer {self.token}'},
                json=payload,
                timeout=30  # Add timeout
            )
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: Status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
            
            if self.delay > 0:
                time.sleep(self.delay)
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling OpenRouter API: {e}")
            return None

    
    def submit_prompt_return_response(self, prompt: str):
        output_data = self.submit_prompt(prompt)
        
        # Handle API failure
        if output_data is None:
            print("OpenRouter API call failed, returning fallback response")
            return "fold"  # Safe fallback action
            
        # Handle malformed response
        try:
            choices = output_data.get('choices')
            if not choices or len(choices) == 0:
                print(f"No choices in OpenRouter response: {output_data}")
                return "fold"
                
            message = choices[0].get('message')
            if not message:
                print(f"No message in OpenRouter choice: {choices[0]}")
                return "fold"
                
            content = message.get('content')
            if content is None:
                print(f"No content in OpenRouter message: {message}")
                return "fold"
                
            return content
            
        except Exception as e:
            print(f"Error parsing OpenRouter response: {e}")
            print(f"Response was: {output_data}")
            return "fold"




