"""
game.py

Utilities for managing the game state in which the LLM agents compete
"""

import random
from typing import List, Optional, Tuple, Dict
from enum import Enum
from dataclasses import dataclass

from src.agent_utils import FeatureReporter
from src.poker_types import ActionType, Action, ActionRecord
from src.core_poker_mechanics import Card, Suit, HandEvaluator

from .logging_config import get_logger
logger = get_logger(__name__)


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
        """Get human-readable stats summary with sample size warnings."""
        if self.hands_played == 0:
            return "No hands played yet"
        
        vpip = self.get_vpip_percent()
        pfr = self.get_pfr_percent()
        agg = self.get_aggression_factor()
        agg_str = f"{agg:.1f}" if agg != float('inf') else "∞"
        
        # Base stats
        base_stats = f"VPIP: {vpip:.1f}% | PFR: {pfr:.1f}% | AGG: {agg_str} | Hands: {self.hands_played}"
        
        # Add sample size warning for small samples
        if self.hands_played < 20:
            warning = f"! Warning: Statistics based on small sample of {self.hands_played} hands!"
            base_stats = f"{warning}\n{base_stats}"
        
        if include_win_rates:
            wr = self.get_win_rate()
            wtsd = self.get_wtsd()
            return f"{base_stats} | WR: {wr:.1f}% | WTSD: {wtsd:.1f}%"
        else:
            return base_stats
        
        
        



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
        self.player_pot_contributions = {p.name: 0 for p in players}
        self.blind_actions = []
        
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
        
        # Track the player's contribution
        self.player_pot_contributions[player.name] += pot_addition
        
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
        
        # DON'T reset player_pot_contributions here - they accumulate for the entire hand
        
        self.last_raiser_index = -1
        # Start with first active player after dealer
        self.current_player_index = 0


class StatsTracker:
    """Track statistics across multiple hands"""
    
    def __init__(self):
        self.player_stats: Dict[str, PlayerStats] = {}
        self.current_hand_vpip = set()
    
    def get_player_stats(self, player_name: str) -> PlayerStats:
        """Get stats for a player, creating if needed"""
        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerStats()
        return self.player_stats[player_name]
    
    def record_hand_start(self, players: List['Player']):
        """Record that a hand has started for all players"""
        self.current_hand_vpip.clear()
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
        
        # Check if this is a blind posting
        is_blind = (action.reasons and 
                    any('blind' in r.lower() for r in action.reasons) and 
                    betting_round == 0)
        
        # Override voluntary flag for blinds
        if is_blind:
            voluntary = False
        
        # Count action types
        if action.action_type == ActionType.FOLD:
            stats.total_folds += 1
        elif action.action_type == ActionType.CHECK:
            stats.total_checks += 1
        elif action.action_type == ActionType.CALL:
            stats.total_calls += 1
            # VPIP: voluntary money in pot (not including blinds)
            if betting_round == 0 and voluntary and player_name not in self.current_hand_vpip:
                if action.action_type in [ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN]:
                    stats.vpip_hands += 1
                    self.current_hand_vpip.add(player_name)
        elif action.action_type in [ActionType.BET, ActionType.ALL_IN]:
            stats.total_bets += 1
            if betting_round == 0 and voluntary and not is_blind:
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
        
        # The GameState __init__ already initializes player_pot_contributions
        # so no additional code needed here
        
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
        """Award pot to winner(s) with proper side pot handling"""
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
            # Showdown - calculate side pots
            result = self._award_with_side_pots(active_players)
            winners = self._get_showdown_winners(active_players)
        
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
    
    def _get_showdown_winners(self, active_players: List['Player']) -> List['Player']:
        """Get list of winners at showdown for stats tracking"""
        results = []
        for player in active_players:
            all_cards = [player.hole_cards[0], player.hole_cards[1]] + self.game_state.board
            rank, tiebreakers = HandEvaluator.evaluate_hand(all_cards)
            results.append((player, rank, tiebreakers))
        
        results.sort(key=lambda x: (x[1].value, x[2]), reverse=True)
        best_rank = results[0][1]
        best_tiebreakers = results[0][2]
        return [r[0] for r in results if r[1] == best_rank and r[2] == best_tiebreakers]
    

    def _award_with_side_pots(self, active_players: list['Player']) -> str:
        """
        Handle pot distribution with proper side pot logic.
        Side pots exist only when at least one all-in player's contribution
        is less than the table's maximum contribution and contribution levels differ.
        """
        contribs = self.game_state.player_pot_contributions
        positive = [c for c in contribs.values() if c > 0]
        if not positive:
            return self._award_simple_main_pot(active_players)
    
        max_contrib = max(positive)
        multiple_levels = len(set(positive)) > 1
    
        def went_all_in(p: 'Player') -> bool:
            if getattr(p, 'is_all_in', False):
                return True
            actions = self.game_state.get_player_actions(p.name)
            return any(a.action.action_type == ActionType.ALL_IN for a in actions)
    
        any_all_in_below_max = any(
            went_all_in(p) and contribs.get(p.name, 0) < max_contrib
            for p in self.players
        )
    
        if multiple_levels and any_all_in_below_max:
            return self._award_complex_side_pots(active_players)
        return self._award_simple_main_pot(active_players)
    
    
    
    def _award_simple_main_pot(self, active_players: List['Player']) -> str:
        """Award a single main pot to the best hand(s)"""
        
        # Evaluate hands
        player_hands = []
        for player in active_players:
            all_cards = [player.hole_cards[0], player.hole_cards[1]] + self.game_state.board
            rank, tiebreakers = HandEvaluator.evaluate_hand(all_cards)
            player_hands.append({
                'player': player,
                'rank': rank,
                'tiebreakers': tiebreakers
            })
        
        # Find winners
        player_hands.sort(key=lambda x: (x['rank'].value, x['tiebreakers']), reverse=True)
        best_rank = player_hands[0]['rank']
        best_tiebreakers = player_hands[0]['tiebreakers']
        
        winners = [ph for ph in player_hands 
                   if ph['rank'] == best_rank and ph['tiebreakers'] == best_tiebreakers]
        
        # Distribute pot
        total_pot = self.game_state.pot
        share_per_winner = total_pot // len(winners)
        remainder = total_pot % len(winners)
        
        result_lines = ["=== POT DISTRIBUTION ==="]
        result_lines.append(f"Main pot {total_pot}:")
        
        for i, winner_info in enumerate(winners):
            award = share_per_winner + (1 if i < remainder else 0)
            winner_info['player'].chips += award
            result_lines.append(f"  {winner_info['player'].name} wins {award}")
        
        return "\n".join(result_lines)
    
    def _award_complex_side_pots(self, active_players: list['Player']) -> str:
        """
        Handle true side pots when players contributed at different capped levels.
        """
        player_hands = []
        for player in active_players:
            all_cards = [player.hole_cards[0], player.hole_cards[1]] + self.game_state.board
            rank, tiebreakers = HandEvaluator.evaluate_hand(all_cards)
            contribution = self.game_state.player_pot_contributions[player.name]
            player_hands.append({
                'player': player,
                'rank': rank,
                'tiebreakers': tiebreakers,
                'contribution': contribution
            })
    
        all_contributions = list(self.game_state.player_pot_contributions.values())
        contribution_levels = sorted(set(c for c in all_contributions if c > 0))
    
        result_lines = ['=== POT DISTRIBUTION ===']
        remaining_pot = self.game_state.pot
        last_level = 0
    
        for level in contribution_levels:
            total_players_at_level = sum(1 for c in all_contributions if c >= level)
            eligible_winners = [ph for ph in player_hands if ph['contribution'] >= level]
    
            pot_increment = level - last_level
            side_pot_size = pot_increment * total_players_at_level
    
            if not eligible_winners:
                remaining_pot -= side_pot_size
                last_level = level
                continue
    
            eligible_sorted = sorted(
                eligible_winners,
                key=lambda x: (x['rank'].value, x['tiebreakers']),
                reverse=True
            )
            best_rank = eligible_sorted[0]['rank']
            best_tiebreakers = eligible_sorted[0]['tiebreakers']
    
            pot_winners = [
                ph for ph in eligible_sorted
                if ph['rank'] == best_rank and ph['tiebreakers'] == best_tiebreakers
            ]
    
            share_per_winner = side_pot_size // len(pot_winners)
            remainder = side_pot_size % len(pot_winners)
    
            if side_pot_size > 0:
                if len(contribution_levels) > 1:
                    result_lines.append(f'Side pot {side_pot_size}:')
                else:
                    result_lines.append(f'Main pot {side_pot_size}:')
    
                for i, winner_info in enumerate(pot_winners):
                    award = share_per_winner + (1 if i < remainder else 0)
                    winner_info['player'].chips += award
                    result_lines.append(f"  {winner_info['player'].name} wins {award}")
    
            remaining_pot -= side_pot_size
            last_level = level
    
        if remaining_pot != 0:
            self.logger.error(f'ERROR: {remaining_pot} chips unaccounted for in complex side pot distribution')
    
        return '\n'.join(result_lines)
    

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
        
        call_amount = max(0, game_state.current_bet - self.current_bet)
        if call_amount > 0:
            pot_odds_ratio = game_state.pot / call_amount
            pot_odds_percent = call_amount / (game_state.pot + call_amount) * 100
            info['pot_odds_ratio'] = pot_odds_ratio
            info['pot_odds_percent'] = pot_odds_percent
            info['break_even_equity'] = pot_odds_percent
        else:
            info['pot_odds_ratio'] = None
            info['pot_odds_percent'] = None
        
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
    
    def _is_blind_defense_situation(self, game_state: GameState) -> bool:
        """Check if this is a blind defense situation (preflop with posted blind)"""
        if game_state.betting_round != 0:  # Only preflop can be blind defense
            return False
        
        # Check if this player posted a blind this hand
        return self.current_bet > 0  # Simple check: if player has chips in pot preflop, they posted a blind
        
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
        
        # After the current betting situation section, add:
        if info['amount_to_call'] > 0:
            pot_odds_ratio = info.get('pot_odds_ratio')
            break_even_equity = info.get('pot_odds_percent')
            lines.append(f"Pot odds: {pot_odds_ratio:.1f}:1 (need {break_even_equity:.1f}% equity to break even)")
            
            # Add bluff assessment for non-blind calls
            if not self._is_blind_defense_situation(game_state):
                lines.append("Consider: Is your opponent likely bluffing based on their betting pattern and board texture?")
            
            # Add blind defense guidance
            if self.current_bet > 0:  # Player has posted a blind
                lines.append(f"You have {self.current_bet} chips already invested as a blind")
                if pot_odds_ratio >= 3.0:  # Getting 3:1 or better
                    lines.append("BLIND DEFENSE: Getting excellent pot odds - fold only very weak hands")
                elif pot_odds_ratio >= 2.0:  # Getting 2:1 or better  
                    lines.append("BLIND DEFENSE: Getting good pot odds - defend with reasonable range")
        
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
        old_dealer = self.dealer_index
        self.dealer_index = self._next_funded_index(self.dealer_index)
        self.logger.info(f"Button moved from position {old_dealer} to {self.dealer_index}")

    def _current_active_seating(self) -> List[Player]:
        """
        Build the in-hand seating list with proper button positioning.
        For heads-up, maintain consistent player positions.
        """
        funded = [p for p in self.players if p.chips > 0]
        if len(funded) <= 1:
            return funded
        
        # For heads-up, maintain table order but track button position
        if len(funded) == 2:
            # Keep players in their original table positions
            return funded
        
        # For 3+ players, order from button
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
        For heads-up, alternate between players; for 3+ players,
        the dealer is always seat 0 in active seating.
        """
        if len(active) == 2:
            # Map table dealer_index to the correct seat in active[]
            # active is built starting from dealer_index, so dealer may be at 0 or 1
            for i, p in enumerate(active):
                if self.players[self.dealer_index] is p:
                    return i
            return 0  # fallback
        else:
            # With 3+, construction guarantees dealer at position 0
            return 0
    
    def _post_blinds(self, hand: Hand) -> None:
        """
        Post small blind and big blind into the hand's GameState before preflop actions.
        """
        gs: GameState = hand.game_state
        players = gs.players
        n = len(players)
        if n < 2:
            return
        
        dealer_pos = gs.dealer_position
        
        if n == 2:
            # Heads-up: Dealer posts SB, other player posts BB
            sb_index = dealer_pos
            bb_index = 1 - dealer_pos  # The other player
            first_to_act = dealer_pos  # Dealer/SB acts first preflop heads-up
        elif n == 3:
            sb_index = (dealer_pos + 1) % n
            bb_index = (dealer_pos + 2) % n
            first_to_act = dealer_pos  # In 3-player, dealer acts first after blinds
        else:
            # 3+ players: standard positions relative to dealer
            sb_index = (dealer_pos + 1) % n
            bb_index = (dealer_pos + 2) % n
            first_to_act = (dealer_pos + 3) % n  # UTG
            
        self.logger.info(
            f"Dealer: {self.players[self.dealer_index].name} "
            f"(SB={self.players[sb_index].name}, BB={self.players[bb_index].name}), "
            f"First to act preflop: {self.players[first_to_act].name}"
        )

        
        # Post SB
        sb_player = players[sb_index]
        sb_amt = min(self.config.small_blind, sb_player.chips)
        sb_action = Action(ActionType.BET, sb_amt, reasons=["small_blind"])
        gs.apply_action(sb_player, sb_action)
        gs.blind_actions.append(len(gs.action_history) - 1)  # Record SB action index
        
        # Post BB
        bb_player = players[bb_index]
        bb_amt = min(self.config.big_blind, bb_player.chips)
        bb_action = Action(ActionType.BET, bb_amt, reasons=["big_blind"])
        gs.apply_action(bb_player, bb_action)
        gs.blind_actions.append(len(gs.action_history) - 1)  # Record BB action index
        
        # Set current bet to the big blind amount
        gs.current_bet = bb_amt
        
        # Set first player to act
        gs.current_player_index = first_to_act
        gs.betting_round = 0

    def _find_next_funded_in_hand(self, players: List[Player], start_index: int) -> int:
        """Find next index (cyclic) with chips > 0 among the already-seated `players` list."""
        n = len(players)
        for i in range(1, n + 1):
            idx = (start_index + i) % n
            if players[idx].chips > 0:
                return idx
        return start_index
    
    