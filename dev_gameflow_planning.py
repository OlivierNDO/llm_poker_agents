import logging
from datetime import datetime
import random
from typing import Any, List, Optional, Tuple, Dict
from enum import Enum
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

from src.core import Card, Suit, HandEvaluator
import src.feature_engineering as fe

load_dotenv()
api_token = os.getenv("NLP_CLOUD_TOKEN")
open_router_api_token = os.getenv("OPEN_ROUTER_TOKEN")





# --- Logging setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"poker_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,  # or DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler()  # keep console output too
    ],
    force = True
)

logger = logging.getLogger()





class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class Action:
    action_type: ActionType
    amount: int = 0
    reasons: Optional[List[str]] = None
    
@dataclass
class ActionRecord:
    """Complete record of an action taken during the hand"""
    player_name: str
    action: Action
    betting_round: int  # 0=preflop, 1=flop, 2=turn, 3=river
    board_at_time: List[Card]  # What the board looked like when action was taken
    pot_before: int
    current_bet_before: int
    position: int  # Position relative to dealer button


@dataclass
class PlayerStats:
    """Track HUD-style statistics for a player"""
    hands_played: int = 0
    hands_won: int = 0
    
    # Preflop stats
    vpip_hands: int = 0  # Voluntarily Put money In Pot (not counting blinds)
    pfr_hands: int = 0   # Pre-Flop Raise
    
    # Aggression stats  
    total_bets: int = 0
    total_calls: int = 0
    total_raises: int = 0
    total_checks: int = 0
    total_folds: int = 0
    
    # Showdown stats
    showdowns_reached: int = 0
    showdowns_won: int = 0
    
    def get_vpip_percent(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return (self.vpip_hands / self.hands_played) * 100
    
    def get_pfr_percent(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return (self.pfr_hands / self.hands_played) * 100
    
    def get_aggression_factor(self) -> float:
        if self.total_calls == 0:
            return float('inf') if (self.total_bets + self.total_raises) > 0 else 0.0
        return (self.total_bets + self.total_raises) / self.total_calls
    
    def get_win_rate(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return (self.hands_won / self.hands_played) * 100
    
    def get_wtsd(self) -> float:
        if self.hands_played == 0:
            return 0.0
        return (self.showdowns_reached / self.hands_played) * 100
    
    def get_readable_stats(self, include_win_rates: bool = False) -> str:
        """Get human-readable stats summary. 
        If include_win_rates=False, hides WR and WTSD to avoid bias leakage.
        """
        if self.hands_played == 0:
            return "No hands played yet"
        
        vpip = self.get_vpip_percent()
        pfr = self.get_pfr_percent()
        agg = self.get_aggression_factor()
        agg_str = f"{agg:.1f}" if agg != float('inf') else "∞"
        
        # Always-safe stats
        base_stats = f"VPIP: {vpip:.1f}% (n={self.hands_played}) | PFR: {pfr:.1f}% (n={self.hands_played}) | AGG: {agg_str} (n={self.hands_played}) | Hands: {self.hands_played}"
        
        
        if include_win_rates:
            wr = self.get_win_rate()
            wtsd = self.get_wtsd()
            return f"{base_stats} | WR: {wr:.1f}% | WTSD: {wtsd:.1f}%"
        else:
            return base_stats



class StatsTracker:
    """Track statistics across multiple hands"""
    
    def __init__(self):
        self.player_stats: Dict[str, PlayerStats] = {}
    
    def get_player_stats(self, player_name: str) -> PlayerStats:
        """Get stats for a player, creating if needed"""
        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerStats()
        return self.player_stats[player_name]
    
    def record_hand_start(self, players: List['Player']):
        """Record that a hand has started for all players"""
        for player in players:
            stats = self.get_player_stats(player.name)
            stats.hands_played += 1
    
    def record_hand_end(self, winners: List['Player'], showdown_players: List['Player']):
        """Record hand results"""
        # Record showdowns
        for player in showdown_players:
            stats = self.get_player_stats(player.name)
            stats.showdowns_reached += 1
            
            if player in winners:
                stats.showdowns_won += 1
        
        # Record wins
        for winner in winners:
            stats = self.get_player_stats(winner.name)
            stats.hands_won += 1
    
    def record_action(self, player_name: str, action: Action, betting_round: int, voluntary: bool = True):
        """Record a player's action for stats"""
        stats = self.get_player_stats(player_name)
        
        # Count action types
        if action.action_type == ActionType.FOLD:
            stats.total_folds += 1
        elif action.action_type == ActionType.CHECK:
            stats.total_checks += 1
        elif action.action_type == ActionType.CALL:
            stats.total_calls += 1
            # VPIP: voluntary money in pot (not including blinds)
            if betting_round == 0 and voluntary:
                stats.vpip_hands += 1
        elif action.action_type in [ActionType.BET, ActionType.ALL_IN]:
            stats.total_bets += 1
            if betting_round == 0 and voluntary:
                stats.vpip_hands += 1
                stats.pfr_hands += 1
        elif action.action_type == ActionType.RAISE:
            stats.total_raises += 1
            if betting_round == 0 and voluntary:
                stats.vpip_hands += 1
                stats.pfr_hands += 1
    
    def get_opponent_stats_summary(self, my_name: str, include_win_rates: bool = False) -> str:
        """Get a summary of all opponents' stats"""
        lines = []
        for player_name, stats in self.player_stats.items():
            if player_name != my_name and stats.hands_played > 0:
                lines.append(f"{player_name}: {stats.get_readable_stats(include_win_rates)}")
        return "\n".join(lines) if lines else "No opponent stats available"



class GameState:
    """Represents the current state of the hand that all players can see"""
    
    def __init__(self, players: List['Player']):
        self.pot = 0
        self.current_bet = 0
        self.board: List[Card] = []
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.players = players
        self.current_player_index = 0
        self.last_raiser_index = -1
        self.action_history: List[ActionRecord] = []  # Complete history of all actions
        self.dealer_position = 0  # Index of dealer button
        
    def get_active_players(self) -> List['Player']:
        return [p for p in self.players if p.is_active]
    
    def get_players_who_can_act(self) -> List['Player']:
        return [p for p in self.players if p.can_act()]
    
    def get_current_player(self) -> Optional['Player']:
        """Get the current player to act"""
        players_who_can_act = self.get_players_who_can_act()
        if not players_who_can_act:
            return None
        
        # Find next player who can act
        for i in range(len(self.players)):
            player_index = (self.current_player_index + i) % len(self.players)
            player = self.players[player_index]
            if player.can_act():
                self.current_player_index = player_index
                return player
        
        return None
    
    def advance_to_next_player(self):
        """Move to next player"""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
    
    def apply_action(self, player: 'Player', action: Action):
        """Apply an action to the game state and record it"""
        # Record the action before applying it
        position = self._get_position(player)
        action_record = ActionRecord(
            player_name=player.name,
            action=action,
            betting_round=self.betting_round,
            board_at_time=self.board.copy(),  # Snapshot of board when action taken
            pot_before=self.pot,
            current_bet_before=self.current_bet,
            position=position
        )
        
        old_current_bet = self.current_bet
        new_current_bet, pot_addition = player.apply_action(action, self)
        
        self.current_bet = new_current_bet
        self.pot += pot_addition
        
        # Track raises
        if new_current_bet > old_current_bet:
            self.last_raiser_index = self.current_player_index
        
        # Add to history after applying
        self.action_history.append(action_record)
    
    def get_action_history(self, betting_round: Optional[int] = None) -> List[ActionRecord]:
        """Get action history, optionally filtered by betting round"""
        if betting_round is None:
            return self.action_history.copy()
        return [a for a in self.action_history if a.betting_round == betting_round]
    
    def get_player_actions(self, player_name: str, betting_round: Optional[int] = None) -> List[ActionRecord]:
        """Get all actions by a specific player"""
        actions = [a for a in self.action_history if a.player_name == player_name]
        if betting_round is not None:
            actions = [a for a in actions if a.betting_round == betting_round]
        return actions
    
    def get_actions_since_board_change(self) -> List[ActionRecord]:
        """Get all actions since the last community card was dealt"""
        current_board = self.board.copy()
        actions_this_round = []
        
        # Go backwards through history until we find a different board
        for action in reversed(self.action_history):
            if action.board_at_time == current_board:
                actions_this_round.append(action)
            else:
                break
        
        return list(reversed(actions_this_round))
    
    def _get_position(self, player: 'Player') -> int:
        """Get player's position relative to dealer (0=dealer, 1=small blind, etc.)"""
        player_index = self.players.index(player)
        return (player_index - self.dealer_position) % len(self.players)
    
    def is_betting_complete(self) -> bool:
        """Check if current betting round is complete"""
        active_players = self.get_active_players()
        if len(active_players) <= 1:
            return True
        
        players_who_can_act = self.get_players_who_can_act()
        if not players_who_can_act:
            return True  # Everyone all-in or folded
        
        # Check if we have any actions this round - if not, betting hasn't started
        actions_this_round = self.get_actions_since_board_change()
        if not actions_this_round:
            return False  # No actions taken yet
        
        # If no raises yet, everyone needs to have acted at least once
        if self.last_raiser_index == -1:
            # Get unique players who have acted this round
            players_who_acted = set(a.player_name for a in actions_this_round)
            active_player_names = set(p.name for p in active_players if p.can_act())
            
            # Everyone who can act must have acted at least once
            return active_player_names.issubset(players_who_acted)
        
        # Someone raised - check if everyone has responded
        for player in active_players:
            if player.can_act() and player.current_bet < self.current_bet:
                return False
        
        return True
    
    def start_new_betting_round(self):
        """Start a new betting round (reset bets for post-flop)"""
        if self.betting_round > 0:  # Don't reset preflop
            for player in self.players:
                player.current_bet = 0
            self.current_bet = 0
        
        self.last_raiser_index = -1
        # Start with first active player after dealer
        self.current_player_index = 0


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

    def __init__(self, hole_cards, game_state : GameState):
        self.hole_cards = hole_cards
        self.game_state = game_state
        self.card_dict = self.get_cards(game_state)
        
        self.hole = [self.parse_card(c) for c in self.card_dict['my_hole_cards']]
        self.board = [self.parse_card(c) for c in self.card_dict['board']]

        self.hole_ranks = np.array([r for r, s in self.hole], dtype=np.int32)
        self.hole_suits = np.array([s for r, s in self.hole], dtype=np.int32)
        self.board_ranks = np.array([r for r, s in self.board], dtype=np.int32)
        self.board_suits = np.array([s for r, s in self.board], dtype=np.int32)
        self.num_opponents = self.game_state.get_active_players()
        
        
    def get_cards(self) -> dict:
        """Get dictionary with player hole cards and board cards"""
        return {
            'my_hole_cards': [card_to_str(c) for c in self.hole_cards],
            'board': [card_to_str(c) for c in self.game_state.board.copy()]
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
    
    def get_cards(self, game_state: GameState) -> dict:
        """Get dictionary with player hole cards and board cards"""
        return {
            'my_hole_cards': [card_to_str(c) for c in self.hole_cards],
            'board': [card_to_str(c) for c in game_state.board.copy()]
        }

    def hand_strength_string(self) -> str:
        """Return a descriptive string with normalized hand strength."""
        
        scalar = fe.extract_hand_strength_scalar(
            hole_ranks = self.hole_ranks,
            hole_suits = self.hole_suits,
            board_ranks = self.board_ranks,
            board_suits = self.board_suits   
        )
        return f"Approximate hand strength {scalar:.1f} (0 to 1 scale)"
    
    def current_hand_string(self) -> str:
        hand_name = fe.best_hand_string(
            hole_ranks = self.hole_ranks,
            hole_suits = self.hole_suits,
            board_ranks = self.board_ranks,
            board_suits = self.board_suits   
        )
        return f"Your current hand is {hand_name}"

    def suited_hole_string(self) -> str:
        """Return a descriptive string indicating whether hole cards are suited or not"""
        if self.hole_suits[0] == self.hole_suits[1]:
            return "Your pocket cards are suited"
        else:
            return "Your pocket cards are not suited"
        
        
    def straight_probability_string(self):
        """Return a descriptive string indicating an approximate probability of getting a straight"""
        
        straight_prob = fe.calculate_straight_probability(
            hole_ranks = self.hole_ranks,
            hole_suits = self.hole_suits,
            board_ranks = self.board_ranks,
            board_suits = self.board_suits
        )
        
        opponent_straight_prob = fe.calculate_opponent_straight_probability(
            hole_ranks = self.hole_ranks,
            hole_suits = self.hole_suits,
            board_ranks = self.board_ranks,
            board_suits = self.board_suits,
            num_opponents = self.num_opponents
        )
        pass
        
    def get_hand_features(self) -> dict:
        """Get descriptive strings about player hand based on feature_engineering.py logic"""
        feature_str_list = []
        if len(self.card_dict.get('board')) > 0:
            # Hand strength scalar string
            #feature_str_list.append(feature_reporter.hand_strength_string())

            # Name of hand
            feature_str_list.append(self.current_hand_string())
            
        if len(self.card_dict.get('board')) == 0:
            # Pocket cards suited
            feature_str_list.append(self.suited_hole_string())

        return feature_str_list


class Player:
    def __init__(self, name: str, chips: int, agent=None, logger = logger):
        self.name = name
        self.chips = chips
        self.agent = agent
        self.logger = logger
        
        # Hand state
        self.hole_cards: Optional[Tuple[Card, Card]] = None
        self.current_bet = 0
        self.is_active = True
        self.is_all_in = False
    
    def reset_for_hand(self):
        self.hole_cards = None
        self.current_bet = 0
        self.is_active = True
        self.is_all_in = False
    
    def can_act(self) -> bool:
        return self.is_active and not self.is_all_in and self.chips > 0
    
    def get_valid_actions(self, game_state: GameState) -> List[ActionType]:
        """Determine valid actions based on game state"""
        if not self.can_act():
            return []
        
        valid_actions = [ActionType.FOLD]
        
        if game_state.current_bet == 0:
            valid_actions.append(ActionType.CHECK)
            if self.chips > 0:
                valid_actions.append(ActionType.BET)
        else:
            call_amount = game_state.current_bet - self.current_bet
            if self.chips >= call_amount:
                valid_actions.append(ActionType.CALL)
                if self.chips > call_amount:
                    valid_actions.append(ActionType.RAISE)
            if self.chips > 0:
                valid_actions.append(ActionType.ALL_IN)
        
        return valid_actions
    
    def apply_action(self, action: Action, game_state: GameState) -> Tuple[int, int]:
        """Apply action and return (new_current_bet, pot_addition)"""
        pot_addition = 0
        new_current_bet = game_state.current_bet
        
        if action.action_type == ActionType.FOLD:
            self.is_active = False
            
        elif action.action_type == ActionType.CHECK:
            pass
            
        elif action.action_type == ActionType.CALL:
            call_amount = game_state.current_bet - self.current_bet
            pot_addition = self._place_bet(call_amount)
            
        elif action.action_type == ActionType.BET:
            pot_addition = self._place_bet(action.amount)
            new_current_bet = self.current_bet
            
        elif action.action_type == ActionType.RAISE:
            call_amount = game_state.current_bet - self.current_bet
            total_amount = call_amount + action.amount
            pot_addition = self._place_bet(total_amount)
            new_current_bet = self.current_bet
            
        elif action.action_type == ActionType.ALL_IN:
            pot_addition = self._place_bet(self.chips)
            if self.current_bet > game_state.current_bet:
                new_current_bet = self.current_bet
        
        return new_current_bet, pot_addition
    
    
    def get_action(self, game_state: GameState, stats_tracker: Optional[StatsTracker] = None) -> Action:
        if not self.agent:
            raise ValueError(f"Player {self.name} has no agent")
    
        valid_actions = self.get_valid_actions(game_state)
        return self.agent.get_action(self, game_state, valid_actions, stats_tracker)
    
    
    def get_cards(self, game_state: GameState) -> dict:
        """Get dictionary with player hole cards and board cards"""
        return {
            'my_hole_cards': [card_to_str(c) for c in self.hole_cards],
            'board': [card_to_str(c) for c in game_state.board.copy()]
        }
    

        
    
    def get_hand_info(self, game_state: GameState) -> dict:
        """Get comprehensive hand information from this player's perspective"""
        
        # Basic hand info
        info = {
            'my_hole_cards': self.hole_cards,
            'my_chips': self.chips,
            'my_current_bet': self.current_bet,
            'pot': game_state.pot,
            'current_bet_to_call': game_state.current_bet,
            'amount_to_call': max(0, game_state.current_bet - self.current_bet),
            'board': game_state.board.copy(),
            'betting_round': game_state.betting_round,
            'betting_round_name': ['PREFLOP', 'FLOP', 'TURN', 'RIVER'][game_state.betting_round]
        }
        
        # Active players info
        active_players = game_state.get_active_players()
        info['active_players'] = []
        info['opponents_remaining'] = 0
        
        for player in active_players:
            player_info = {
                'name': player.name,
                'chips': player.chips,
                'current_bet': player.current_bet,
                'is_all_in': player.is_all_in,
                'position': game_state._get_position(player)
            }
            info['active_players'].append(player_info)
            
            if player.name != self.name:
                info['opponents_remaining'] += 1
        
        # Action history this round
        actions_this_round = game_state.get_actions_since_board_change()
        info['actions_this_round'] = []
        
        for action_record in actions_this_round:
            action_info = {
                'player': action_record.player_name,
                'action': action_record.action.action_type.value,
                'amount': action_record.action.amount or 0,
                'pot_before': action_record.pot_before,
                'position': action_record.position
            }
            info['actions_this_round'].append(action_info)
        
        # Full hand history (all betting rounds)
        info['full_hand_history'] = []
        for action_record in game_state.action_history:
            round_name = ['PREFLOP', 'FLOP', 'TURN', 'RIVER'][action_record.betting_round]
            action_info = {
                'round': round_name,
                'player': action_record.player_name,
                'action': action_record.action.action_type.value,
                'amount': action_record.action.amount or 0,
                'board_at_time': action_record.board_at_time.copy(),
                'position': action_record.position
            }
            info['full_hand_history'].append(action_info)
        
        # My actions this hand
        my_actions = game_state.get_player_actions(self.name)
        info['my_actions_this_hand'] = []
        for action_record in my_actions:
            round_name = ['PREFLOP', 'FLOP', 'TURN', 'RIVER'][action_record.betting_round]
            info['my_actions_this_hand'].append({
                'round': round_name,
                'action': action_record.action.action_type.value,
                'amount': action_record.action.amount or 0
            })
        
        # Specific opponent actions (useful for reads)
        info['opponent_actions'] = {}
        for player in active_players:
            if player.name != self.name:
                opponent_actions = game_state.get_player_actions(player.name)
                info['opponent_actions'][player.name] = []
                for action_record in opponent_actions:
                    round_name = ['PREFLOP', 'FLOP', 'TURN', 'RIVER'][action_record.betting_round]
                    info['opponent_actions'][player.name].append({
                        'round': round_name,
                        'action': action_record.action.action_type.value,
                        'amount': action_record.action.amount or 0,
                        'board_at_time': action_record.board_at_time.copy()
                    })
        
        return info
    
    def get_readable_hand_summary(self, game_state: GameState) -> str:
        """Get a human-readable summary of the hand situation"""
        info = self.get_hand_info(game_state)
        
        lines = []
        lines.append(f"=== {self.name}'s Hand Summary ===")
        lines.append(f"My cards: {info['my_hole_cards'][0]} {info['my_hole_cards'][1]}")
        lines.append(f"My chips: {info['my_chips']}, My bet: {info['my_current_bet']}")
        lines.append(f"Pot: {info['pot']}, Current bet to call: {info['current_bet_to_call']}")
        lines.append(f"I need to call: {info['amount_to_call']}")
        lines.append(f"Board: {' '.join(str(c) for c in info['board']) if info['board'] else 'No community cards yet'}")
        lines.append(f"Betting round: {info['betting_round_name']}")
        lines.append(f"Opponents remaining: {info['opponents_remaining']}")
        
        lines.append(f"\nActions this round:")
        if info['actions_this_round']:
            for action in info['actions_this_round']:
                amt = action.get('amount') or 0
                amount_str = f" {amt}" if amt > 0 else ""
                lines.append(f"  {action['player']}: {action['action']}{amount_str}")
        else:
            lines.append("  No actions yet this round")
        
        lines.append(f"\nActive players:")
        for player_info in info['active_players']:
            status = " (ALL-IN)" if player_info['is_all_in'] else ""
            lines.append(f"  {player_info['name']}: {player_info['chips']} chips, bet {player_info['current_bet']}{status}")
        
        return "\n".join(lines)
    
    def get_llm_prompt_info(self, game_state: GameState, stats_tracker: Optional[StatsTracker] = None) -> str:
        """Get hand information formatted for LLM prompts (without hand features - those are requested separately)"""
        info = self.get_hand_info(game_state)
        
        lines = []
        
        # Basic situation
        lines.append(f"You are {self.name}, an AI poker agent competing in a Texas Hold'em hand.")
        lines.append("All opponents at the table are also AI agents who receive similar information.")
        lines.append("Your objective is to maximize your total chip winnings over many hands, not just in a single hand.")
        lines.append(f"Your hole cards: {info['my_hole_cards'][0]} {info['my_hole_cards'][1]}")
        lines.append(f"Your chips: {info['my_chips']}")
        lines.append(f"Current pot: {info['pot']}")
        
        # Board state
        if info['board']:
            board_str = " ".join(str(card) for card in info['board'])
            lines.append(f"Community cards: {board_str}")
        else:
            lines.append("Community cards: None dealt yet")
            
        # NOTE: Hand features removed - these are now only added when specifically requested via need_hand_analysis
            
        lines.append(f"Betting round: {info['betting_round_name']}")
        
        # Current betting situation
        if info['amount_to_call'] > 0:
            lines.append(f"You need to call {info['amount_to_call']} to stay in the hand")
        else:
            lines.append("No bet to call - you can check or bet")
        
        lines.append(f"Your current bet this round: {info['my_current_bet']}")
        
        # Opponents
        lines.append(f"\nOpponents remaining: {info['opponents_remaining']}")
        for player_info in info['active_players']:
            if player_info['name'] != self.name:
                status = " (ALL-IN)" if player_info['is_all_in'] else ""
                lines.append(f"  {player_info['name']}: {player_info['chips']} chips, bet {player_info['current_bet']} this round{status}")
        
        # Add opponent stats if available
        if stats_tracker:
            opponent_stats = stats_tracker.get_opponent_stats_summary(self.name)
            if opponent_stats:
                lines.append(f"\nOpponent playing styles:")
                lines.append(opponent_stats)
        
        # Action sequence this round
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
        
        # Previous rounds (if any)
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

    def get_hand_features(self, game_state: GameState) -> List[str]:
        """Get descriptive hand features (separated from basic prompt info)"""
        try:
            feature_reporter = FeatureReporter(self.hole_cards, game_state)
            return feature_reporter.get_hand_features()
        except Exception as e:
            self.logger.warning(f"Hand feature extraction failed for {self.name}: {e}")
            return ["Hand feature analysis unavailable"]
    
    def get_simple_situation(self, game_state: GameState) -> str:
        """Get a very concise situation summary"""
        info = self.get_hand_info(game_state)
        
        # My cards and chips
        cards = f"{info['my_hole_cards'][0]} {info['my_hole_cards'][1]}"
        
        # Action summary
        action_summary = []
        for action in info['actions_this_round']:
            amt = action.get('amount') or 0
            if amt > 0:
                action_summary.append(f"{action['player']} {action['action']} {amt}")
            else:
                action_summary.append(f"{action['player']} {action['action']}")
        
        actions_str = ", ".join(action_summary) if action_summary else "No actions yet"
        
        # What I need to do
        if info['amount_to_call'] > 0:
            decision = f"Need to call {info['amount_to_call']}"
        else:
            decision = "Can check or bet"
        
        return f"{self.name}: {cards} | {info['my_chips']} chips | Pot: {info['pot']} | {actions_str} | {decision}"
    
    def _place_bet(self, amount: int) -> int:
        """Place bet and return actual amount"""
        actual_bet = min(amount, self.chips)
        self.chips -= actual_bet
        self.current_bet += actual_bet
        
        if self.chips == 0:
            self.is_all_in = True
        
        return actual_bet
    

class Hand:
    """Manages a complete poker hand from start to finish"""
    
    def __init__(self, players: List['Player'], stats_tracker: Optional[StatsTracker] = None, logger = logger):
        self.players = players
        self.deck = Deck()
        self.game_state = GameState(players)
        self.stats_tracker = stats_tracker
        self.logger = logger
        
        # Hand progression state
        self.phase = "setup"  # setup -> hole_cards -> preflop -> flop -> flop_betting -> turn -> turn_betting -> river -> river_betting -> showdown -> complete
        self.is_complete = False
        
    def execute_next_step(self) -> str:
        """Execute the next step in the hand and return description of what happened"""
        if self.is_complete:
            return "Hand already complete"
        
        if self.phase == "setup":
            return self._setup_hand()
        elif self.phase == "hole_cards":
            return self._deal_hole_cards()
        elif self.phase == "preflop":
            return self._step_betting_round()
        elif self.phase == "flop":
            return self._deal_flop()
        elif self.phase == "flop_betting":
            return self._step_betting_round()
        elif self.phase == "turn":
            return self._deal_turn()
        elif self.phase == "turn_betting":
            return self._step_betting_round()
        elif self.phase == "river":
            return self._deal_river()
        elif self.phase == "river_betting":
            return self._step_betting_round()
        elif self.phase == "showdown":
            return self._showdown()
        elif self.phase == "complete":
            return self._award_pot()
        else:
            return "Unknown phase"
    
    def get_hand_status(self) -> str:
        """Get current status of the hand"""
        active_count = len(self.game_state.get_active_players())
        return f"Phase: {self.phase}, Active players: {active_count}, Pot: {self.game_state.pot}, Current bet: {self.game_state.current_bet}"
    
    def play(self):
        """Play a complete hand (original method for convenience)"""
        while not self.is_complete:
            result = self.execute_next_step()
            self.logger.info(result)
        
    def _setup_hand(self) -> str:
        """Reset everything for new hand"""
        self.deck.reset()
        for player in self.players:
            player.reset_for_hand()
        self.game_state = GameState(self.players)
        
        # Record hand start for stats
        if self.stats_tracker:
            self.stats_tracker.record_hand_start(self.players)
        
        self.phase = "hole_cards"
        return "=== NEW HAND SETUP ==="
    
    def _deal_hole_cards(self) -> str:
        """Deal hole cards to all players"""
        result = []
        for player in self.players:
            player.hole_cards = (self.deck.deal(), self.deck.deal())
            result.append(f"{player.name} dealt: {player.hole_cards[0]} {player.hole_cards[1]}")
        
        self.phase = "preflop"
        self.game_state.betting_round = 0
        return "\n".join(result)
    
    def _deal_flop(self) -> str:
        """Deal flop"""
        self.deck.deal()  # Burn
        flop = [self.deck.deal() for _ in range(3)]
        self.game_state.board.extend(flop)
        self.game_state.betting_round = 1
        self.game_state.start_new_betting_round()
        self.phase = "flop_betting"
        return f"Flop: {' '.join(str(c) for c in flop)}"
    
    def _deal_turn(self) -> str:
        """Deal turn"""
        self.deck.deal()  # Burn
        turn = self.deck.deal()
        self.game_state.board.append(turn)
        self.game_state.betting_round = 2
        self.game_state.start_new_betting_round()
        self.phase = "turn_betting"
        return f"Turn: {turn} (Board: {' '.join(str(c) for c in self.game_state.board)})"
    
    def _deal_river(self) -> str:
        """Deal river"""
        self.deck.deal()  # Burn
        river = self.deck.deal()
        self.game_state.board.append(river)
        self.game_state.betting_round = 3
        self.game_state.start_new_betting_round()
        self.phase = "river_betting"
        return f"River: {river} (Board: {' '.join(str(c) for c in self.game_state.board)})"
    
    def _step_betting_round(self) -> str:
        """Execute one betting action or complete the betting round"""
        round_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
        round_name = round_names[self.game_state.betting_round]
        
        # Check if betting is complete
        if self.game_state.is_betting_complete():
            return self._advance_to_next_phase()
        
        # Get current player
        current_player = self.game_state.get_current_player()
        if not current_player:
            return self._advance_to_next_phase()
        
        # Execute one action
        valid_actions = current_player.get_valid_actions(self.game_state)
        #action = current_player.get_action(self.game_state)
        action = current_player.get_action(self.game_state, self.stats_tracker)

        
        # Record action for stats
        if self.stats_tracker:
            # Determine if this is a voluntary action (not blinds)
            voluntary = not (self.game_state.betting_round == 0 and len(self.game_state.action_history) < 2)
            self.stats_tracker.record_action(current_player.name, action, self.game_state.betting_round, voluntary)
        
        self.game_state.apply_action(current_player, action)
        self.game_state.advance_to_next_player()
        
        action_str = f"{action.action_type.value}"
        if action.amount and action.amount > 0:
            action_str += f" {action.amount}"
        
        return f"{round_name}: {current_player.name} {action_str} (Pot: {self.game_state.pot}, Current bet: {self.game_state.current_bet})"
    
    def _advance_to_next_phase(self) -> str:
        """Move to the next phase of the hand"""
        active_players = self.game_state.get_active_players()
        
        # Check if hand is over (only one player left)
        if len(active_players) <= 1:
            self.phase = "complete"
            return "Betting complete - hand over (fold out)"
        
        # Advance to next phase
        if self.phase == "preflop":
            self.phase = "flop"
            return "Preflop betting complete"
        elif self.phase == "flop_betting":
            self.phase = "turn"
            return "Flop betting complete"
        elif self.phase == "turn_betting":
            self.phase = "river"
            return "Turn betting complete"
        elif self.phase == "river_betting":
            self.phase = "showdown"
            return "River betting complete"
        
        return "Phase transition"
    
    def _showdown(self) -> str:
        """Evaluate hands at showdown"""
        result = ["=== SHOWDOWN ==="]
        active_players = self.game_state.get_active_players()
        
        for player in active_players:
            all_cards = [player.hole_cards[0], player.hole_cards[1]] + self.game_state.board
            rank, tiebreakers = HandEvaluator.evaluate_hand(all_cards)
            best_five = HandEvaluator._find_best_five_cards(all_cards)
            description = HandEvaluator._create_readable_description(rank, tiebreakers, best_five)
            result.append(f"{player.name}: {player.hole_cards[0]} {player.hole_cards[1]} - {description}")
        
        self.phase = "complete"
        return "\n".join(result)
    
    def _award_pot(self) -> str:
        """Award pot to winner(s)"""
        active_players = self.game_state.get_active_players()
        winners = []  # Initialize winners list
        
        if len(active_players) <= 1:
            # Winner by elimination
            if active_players:
                winner = active_players[0]
                winner.chips += self.game_state.pot
                winners = [winner]  # Set winners for stats tracking
                result = f"{winner.name} wins {self.game_state.pot} (others folded)"
            else:
                result = "No winner (shouldn't happen)"
        else:
            # Showdown - find best hand(s)
            results = []
            for player in active_players:
                all_cards = [player.hole_cards[0], player.hole_cards[1]] + self.game_state.board
                rank, tiebreakers = HandEvaluator.evaluate_hand(all_cards)
                results.append((player, rank, tiebreakers))
            
            results.sort(key=lambda x: (x[1].value, x[2]), reverse=True)
            best_rank = results[0][1]
            best_tiebreakers = results[0][2]
            winners = [r[0] for r in results if r[1] == best_rank and r[2] == best_tiebreakers]
            
            # Split pot
            pot_share = self.game_state.pot // len(winners)
            remainder = self.game_state.pot % len(winners)
            
            result_lines = [f"Awarding pot of {self.game_state.pot}:"]
            for i, winner in enumerate(winners):
                share = pot_share + (1 if i < remainder else 0)
                winner.chips += share
                result_lines.append(f"  {winner.name} wins {share}")
            
            result = "\n".join(result_lines)
        
        # Show final chip counts
        chip_lines = ["Final chip counts:"]
        for player in self.players:
            chip_lines.append(f"  {player.name}: {player.chips}")
        
        self.is_complete = True
        
        # Record hand end for stats
        if self.stats_tracker:
            showdown_players = active_players if len(active_players) > 1 else []
            self.stats_tracker.record_hand_end(winners, showdown_players)
        
        return result + "\n" + "\n".join(chip_lines)


class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        self.cards = []
        for suit in Suit:
            for rank in range(2, 15):
                self.cards.append(Card(rank, suit))
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self) -> Card:
        return self.cards.pop()


class Agent(ABC):
    @abstractmethod
    def get_action(self, player: Player, game_state: GameState, valid_actions: List[ActionType]) -> Action:
        pass


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
    
    
@dataclass
class TableConfig:
    """Simple table configuration."""
    small_blind: int = 1
    big_blind: int = 2
    max_hands: Optional[int] = None  # None = infinite
    
    
    
class TableManager:
    """
    Manage multi-hand gameplay: seating, blinds, button rotation, and eliminations.

    Parameters
    ----------
    players : list[Player]
        Initial seating order (index 0 is seat 0).
    config : TableConfig
        Blind sizes and optional max hand count.
    stats_tracker : StatsTracker | None
        Optional HUD-style stats aggregation.

    Notes
    -----
    - Only players with chips > 0 are seated for each new hand.
    - Dealer button rotates to the next remaining player (skips busted seats).
    - Posts blinds into the Hand/GameState so your existing betting code works.
    - Stops when <= 1 player has chips or max_hands is reached.
    """

    def __init__(self, players: List[Player], config: Optional[TableConfig] = None, stats_tracker: Optional[StatsTracker] = None, logger = logger):
        self.players: List[Player] = players
        self.config = config or TableConfig()
        self.stats_tracker = stats_tracker
        self.logger = logger
        self.hand_no = 0
        # Start with dealer at seat 0
        self.dealer_index = 0

    def play(self) -> None:
        """Run hands until one player remains or `max_hands` reached."""
        while self._count_funded_players() > 1:
            if self.config.max_hands is not None and self.hand_no >= self.config.max_hands:
                break

            active_seating = self._current_active_seating()
            if len(active_seating) <= 1:
                break

            # Create a Hand with the funded players in seat order
            hand = Hand(active_seating, self.stats_tracker)

            # Inform GameState where the button is (relative to hand.players list)
            hand.game_state.dealer_position = self._dealer_pos_in_active(active_seating)

            # Deal hole cards first (uses your existing method)
            self.logger.info(hand.execute_next_step())  # setup
            self.logger.info(hand.execute_next_step())  # hole_cards

            # Post SB/BB into the preflop betting round
            self._post_blinds(hand)

            # Progress hand to completion using your existing state machine
            while not hand.is_complete:
                self.logger.info(hand.execute_next_step())

            # Hand finished
            self.logger.info('\n\n\n\n\n\n--------------------------------\n\n\n\n\n\n')
            self.hand_no += 1
            self._advance_button()

            # Eliminate broke players from the table going forward
            # (they remain in self.players list but are simply unseated for future hands)
            # No-op here; seating is derived each hand from chips > 0

            # If only one funded player remains, stop
            if self._count_funded_players() <= 1:
                break

    # ---------- Internals ----------

    def _count_funded_players(self) -> int:
        return sum(1 for p in self.players if p.chips > 0)

    def _next_funded_index(self, start_index: int) -> int:
        """Find next index (cyclic) with chips > 0; returns start_index if none found (edge)."""
        n = len(self.players)
        for i in range(1, n + 1):
            idx = (start_index + i) % n
            if self.players[idx].chips > 0:
                return idx
        return start_index

    def _advance_button(self) -> None:
        """Move dealer button to next funded seat."""
        self.dealer_index = self._next_funded_index(self.dealer_index)

    def _current_active_seating(self) -> List[Player]:
        """
        Build the in-hand seating list (button-relative) from table players who have chips.

        Returns
        -------
        list[Player]
            Ordered starting from dealer seat (inclusive), wrapping around.
        """
        funded = [(i, p) for i, p in enumerate(self.players) if p.chips > 0]
        if not funded:
            return []

        # Order seats from dealer_index, wrapping around table
        ordered: List[Player] = []
        n = len(self.players)
        for offset in range(n):
            idx = (self.dealer_index + offset) % n
            if self.players[idx].chips > 0:
                ordered.append(self.players[idx])
        return ordered

    def _dealer_pos_in_active(self, active: List[Player]) -> int:
        """
        Compute dealer position within the `active` list.

        The active list is built starting at the current dealer_index, so the dealer
        is always at position 0 in the active seating.
        """
        # By construction in _current_active_seating(), dealer is at position 0
        return 0

    def _post_blinds(self, hand: Hand) -> None:
        """
        Post small blind and big blind into the hand's GameState before preflop actions.

        Notes
        -----
        - Handles short stacks by capping bet via Player._place_bet (called inside apply_action).
        - Marks the correct current player to act (UTG) after BB posts.
        - Records actions in GameState history; StatsTracker will treat these as non-voluntary
          because your dev code uses the first two actions as blinds. :contentReference[oaicite:1]{index=1}
        """
        gs: GameState = hand.game_state
        players = gs.players
        n = len(players)
        if n < 2:
            return

        # Dealer is at position 0 in `players`; SB is next funded, then BB next funded after SB
        sb_index = self._find_next_funded_in_hand(players, 0)
        bb_index = self._find_next_funded_in_hand(players, sb_index)

        # Post SB
        sb_player = players[sb_index]
        sb_amt = self.config.small_blind
        sb_action = Action(ActionType.BET, sb_amt)
        gs.apply_action(sb_player, sb_action)

        # Post BB
        bb_player = players[bb_index]
        bb_amt = self.config.big_blind
        bb_action = Action(ActionType.BET, bb_amt)
        gs.apply_action(bb_player, bb_action)

        # Action starts UTG: first funded player after BB
        gs.current_player_index = self._find_next_funded_in_hand(players, bb_index)

        # Ensure preflop betting_round is set
        gs.betting_round = 0

    def _find_next_funded_in_hand(self, players: List[Player], start_index: int) -> int:
        """Find next index (cyclic) with chips > 0 among the already-seated `players` list."""
        n = len(players)
        for i in range(1, n + 1):
            idx = (start_index + i) % n
            if players[idx].chips > 0:
                return idx
        return start_index
    
    



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

    def submit_prompt(self, prompt: str, max_tokens: int | None = 40000, temperature: float = 0.7):
        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
        }
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
    
        response = requests.post(
            self.url,
            headers={'Authorization': f'Bearer {self.token}'},
            json=payload,
        )
        return response.json()

    
    def submit_prompt_return_response(self, prompt: str):
        output_data = self.submit_prompt(prompt)
        output_string = output_data.get('choices')[0].get('message').get('content')
        return output_string






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
                 verbose: bool | int = True,
                 logger = logger):
        self.name = name
        self.player_profile = player_profile or (
            'You are an AI poker agent playing Texas Hold\'em against other agents. '
            'Make rational decisions based on available information. Use planning to gather '
            'relevant data before making betting decisions.'
        )
        self.client = client
        self.verbose = bool(verbose)
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
        
        raw_text = self.client.submit_prompt_return_response(planning_prompt)
        raw_text = (raw_text or '').strip()
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
                self.logger.info(f"{self.name}: Gathered opponent stats: {len(opponent_stats.split(chr(10)))} opponent(s) analyzed")
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
        if info['amount_to_call'] > 0:
            lines.append(f"You need to call {info['amount_to_call']} to stay in the hand")
        else:
            lines.append("No bet to call - you can check or bet")
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
        
        
    def _make_betting_decision(
        self,
        player: Player,
        game_state: GameState,
        valid_actions: List[ActionType],
        gathered_info: dict
    ) -> Action:
        """Make betting decision with improved prompt for consistent JSON output"""
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
        info_sections = [gathered_info["basic_info"]]
        if "opponent_stats" in gathered_info:
            info_sections.append(f"\nOPPONENT STATISTICS:\n{gathered_info['opponent_stats']}")
        if "hand_analysis" in gathered_info:
            analysis_str = "\n".join(gathered_info["hand_analysis"])
            info_sections.append(f"\nHAND ANALYSIS:\n{analysis_str}")
        combined_info = "\n".join(info_sections)
        
        decision_prompt = f"""{self.player_profile}
    
    You are making a poker decision. Your response must be a single JSON object with no additional text, explanations, or formatting.
    
    SITUATION:
    {combined_info}
    
    LEGAL ACTIONS: {legal_str}
    CONSTRAINTS:
    - If action is "bet": amount must be 1-{max_bet}
    - If action is "raise": amount must be 1-{max_raise}  
    - If action is "check", "call", "fold", "all_in": amount must be null
    
    Your response must start with {{ and end with }} and contain only this JSON structure:
    {{
      "action": "ACTION_HERE",
      "amount": AMOUNT_OR_NULL,
      "reasons": ["reason1", "reason2"]
    }}
    
    Do not include any other text. Do not use markdown formatting. Do not use code blocks. Do not add explanations before or after the JSON. Your entire response must be exactly the JSON object and nothing else.
    
    Start your response with the opening brace:"""
        
        raw_text = self.client.submit_prompt_return_response(decision_prompt)
        raw_text = (raw_text or '').strip()
        
        if self.verbose:
            self.logger.info(f"{self.name}: Raw decision response: '{raw_text[:200]}...'")
        
        # Try parsing the response directly
        try:
            decision = json.loads(raw_text)
            if self.verbose:
                self.logger.info(f"{self.name}: Successfully parsed JSON on first try")
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"{self.name}: Initial JSON parsing failed: {e}")
            decision = self._repair_decision_json(raw_text, legal_str, max_bet, max_raise)
        
        return self._validate_and_create_action(
            decision, allowed_actions, call_needed, max_bet, max_raise, player
        )
    
   
    
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










    
open_ai_gpt_4o_client = OpenRouterCompletionEngine(model='openai/gpt-4o')
anthropic_claud_sonnet_4_client = OpenRouterCompletionEngine(model='anthropic/claude-sonnet-4')
google_gemini_flash_2_5_client = OpenRouterCompletionEngine(model='google/gemini-2.5-flash')





player_open_ai_gpt_4o = Player(name='Open Ai Gpt-4o', chips=100, agent=LLMAgent(client = open_ai_gpt_4o_client, name='Open Ai Gpt-4o'))
player_anthropic_claud_sonnet_4_client = Player(name='Claude Sonnet 4', chips=100, agent=LLMAgent(client = anthropic_claud_sonnet_4_client, name='Claude Sonnet 4'))
player_google_gemini_flash_2_5 = Player(name='Gemini Flash 2.5', chips=100, agent=LLMAgent(client = google_gemini_flash_2_5_client, name='Gemini Flash 2.5'))



# Make some players
players = [
    player_open_ai_gpt_4o,
    player_anthropic_claud_sonnet_4_client,
    player_google_gemini_flash_2_5
    
]

# Optional tracker for HUD-style stats
stats = StatsTracker()

# Configure table
#config = TableConfig(small_blind=1, big_blind=2, max_hands=300)

config = TableConfig(small_blind=2, big_blind=3, max_hands=200)

# Create the table manager
tm = TableManager(players, config, stats)

# Run hands until stop condition (bust-out or max_hands)
tm.play()

# Afterward, check results
for p in players:
    logger.info(f'{p.name}: {p.chips} chips')
    
    

    
"""    
    
players = [alice, bob]

# Create a stats tracker if you want
stats = StatsTracker()

# Set up a hand
hand = Hand(players, stats)

# Step 1: Setup
print(hand.execute_next_step())  # "=== NEW HAND SETUP ==="

# Step 2: Deal hole cards
print(hand.execute_next_step())

#print('\n\n\n\n')
#print(alice.get_llm_prompt_info(hand.game_state, hand.stats_tracker))
#print('\n\n\n\n')
# Alice bet
print(hand.execute_next_step())
    
    
# Bob bet    
print(hand.execute_next_step())






    
    
# Flop dealt
print(hand.execute_next_step())
print(hand.execute_next_step())


#print('\n\n\n\n')
#print(alice.get_llm_prompt_info(hand.game_state, hand.stats_tracker))
#print('\n\n\n\n')


# Post flop betting
print(hand.execute_next_step()) # Alice Bet

print(hand.execute_next_step()) # Bob Bet
print(hand.execute_next_step())



# Deal turn card
print(hand.execute_next_step())


#print('\n\n\n\n')
#print(alice.get_llm_prompt_info(hand.game_state, hand.stats_tracker))
#print('\n\n\n\n')


# Post turn betting
print(hand.execute_next_step())
print(hand.execute_next_step())
print(hand.execute_next_step())



# Deal river card
print(hand.execute_next_step())



#print('\n\n\n\n')
#print(alice.get_llm_prompt_info(hand.game_state, hand.stats_tracker))
#print('\n\n\n\n')


print(hand.execute_next_step()) # Alice bets

#print('\n\n\n\n')
#print(bob.get_llm_prompt_info(hand.game_state, hand.stats_tracker))
#print('\n\n\n\n')


print(hand.execute_next_step()) # Bob bets



#print('\n\n\n\n')
#print(alice.get_llm_prompt_info(hand.game_state, hand.stats_tracker))
#print('\n\n\n\n')


print(hand.execute_next_step())












"""


"""
alice.get_cards(hand.game_state)

{'my_hole_cards': (Card(rank=10, suit=<Suit.SPADES: 3>),
  Card(rank=6, suit=<Suit.HEARTS: 2>)),
 'board': [Card(rank=4, suit=<Suit.HEARTS: 2>),
  Card(rank=12, suit=<Suit.CLUBS: 0>),
  Card(rank=14, suit=<Suit.DIAMONDS: 1>)]}




alice.get_hand_info(hand.game_state)

    
print(alice.get_llm_prompt_info(hand.game_state, hand.stats_tracker))
    

player_alice = Player(name='Alice', chips=100, agent=LLMAgent())
player_adrian = Player(name='Adrian', chips=100, agent=LLMAgent())
player_bob = Player(name='Bob', chips=100, agent=RandomAgent())
#player_charlie = Player(name='Charlie', chips=100, agent=AnalyticalAgent())
    
# Make some players
players = [
    player_alice,
    player_adrian,
    player_bob,
    #player_charlie
    
]

# Optional tracker for HUD-style stats
stats = StatsTracker()

# Configure table
#config = TableConfig(small_blind=1, big_blind=2, max_hands=300)

config = TableConfig(small_blind=4, big_blind=5, max_hands=300)

# Create the table manager
tm = TableManager(players, config, stats)

# Run hands until stop condition (bust-out or max_hands)
tm.play()

# Afterward, check results
for p in players:
    print(f'{p.name}: {p.chips} chips')
    
"""

