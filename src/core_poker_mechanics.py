import random
import itertools
from dataclasses import dataclass
from collections import Counter
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class Suit(Enum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

@dataclass
class Card:
    rank: int  # 2-14 (2-10, J=11, Q=12, K=13, A=14)
    suit: Suit

    def __str__(self):
        rank_str = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}.get(self.rank, str(self.rank))
        suit_str = {Suit.CLUBS: '♣', Suit.DIAMONDS: '♦', Suit.HEARTS: '♥', Suit.SPADES: '♠'}[self.suit]
        return f"{rank_str}{suit_str}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

class GameState:
    def __init__(self, player_hands: List[Tuple[Card, Card]], board: List[Card], num_players: int):
        self.player_hands = player_hands
        self.board = board
        self.num_players = num_players

    def __str__(self):
        result = f"Players: {self.num_players}\n"
        for i, hand in enumerate(self.player_hands):
            if hand is not None:
                card1, card2 = hand
                result += f"Player {i}: {card1} {card2}\n"
            else:
                result += f"Player {i}: [Hidden]\n"
        board_str = " ".join(str(card) for card in self.board) if self.board else "Empty"
        result += f"Board: {board_str}\n"
        return result

class HandRank(Enum):
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


        
class HandEvaluator:
    @staticmethod
    def _evaluate_5_cards(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """Evaluate exactly 5 cards"""
        ranks = [card.rank for card in cards]
        suits = [card.suit for card in cards]

        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)

        # Check for flush
        is_flush = len(suit_counts) == 1

        # Check for straight
        sorted_ranks = sorted(set(ranks))
        is_straight = False
        straight_high = 0

        # Regular straight
        if len(sorted_ranks) == 5 and sorted_ranks[-1] - sorted_ranks[0] == 4:
            is_straight = True
            straight_high = sorted_ranks[-1]
        # Ace-low straight (A,2,3,4,5)
        elif sorted_ranks == [2, 3, 4, 5, 14]:
            is_straight = True
            straight_high = 5  # In ace-low straight, 5 is the high card

        # Get counts in descending order
        count_values = sorted(rank_counts.values(), reverse=True)

        # Straight Flush
        if is_straight and is_flush:
            return HandRank.STRAIGHT_FLUSH, [straight_high]

        # Four of a Kind - FIXED
        if count_values == [4, 1]:
            quad_rank = max([rank for rank, count in rank_counts.items() if count == 4])
            kicker = max([rank for rank, count in rank_counts.items() if count == 1])  # FIXED: Use max()
            return HandRank.FOUR_OF_A_KIND, [quad_rank, kicker]

        # Full House - FIXED
        if count_values == [3, 2]:
            trips_rank = max([rank for rank, count in rank_counts.items() if count == 3])
            pair_rank = max([rank for rank, count in rank_counts.items() if count == 2])
            return HandRank.FULL_HOUSE, [trips_rank, pair_rank]

        # Flush
        if is_flush:
            return HandRank.FLUSH, sorted(ranks, reverse=True)

        # Straight
        if is_straight:
            return HandRank.STRAIGHT, [straight_high]

        # Three of a Kind - FIXED
        if count_values == [3, 1, 1]:
            trips_rank = max([rank for rank, count in rank_counts.items() if count == 3])
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return HandRank.THREE_OF_A_KIND, [trips_rank] + kickers

        # Two Pair - FIXED
        if count_values == [2, 2, 1]:
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            kicker = max([rank for rank, count in rank_counts.items() if count == 1])  # FIXED: Use max()
            return HandRank.TWO_PAIR, pairs + [kicker]

        # One Pair - FIXED
        if count_values == [2, 1, 1, 1]:
            pair_rank = max([rank for rank, count in rank_counts.items() if count == 2])
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return HandRank.ONE_PAIR, [pair_rank] + kickers

        # High Card
        return HandRank.HIGH_CARD, sorted(ranks, reverse=True)

    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """
        Evaluate a 7-card hand and return (rank, tiebreakers)
        Tiebreakers are in descending order of importance
        """
        if len(cards) != 7:
            raise ValueError("Hand evaluation requires exactly 7 cards")

        # Find the best 5-card hand from 7 cards
        best_rank = HandRank.HIGH_CARD
        best_tiebreakers = []

        for combo in itertools.combinations(cards, 5):
            rank, tiebreakers = HandEvaluator._evaluate_5_cards(list(combo))
            if rank.value > best_rank.value or (rank == best_rank and tiebreakers > best_tiebreakers):
                best_rank = rank
                best_tiebreakers = tiebreakers

        return best_rank, best_tiebreakers

    @staticmethod
    def _find_best_five_cards(cards: List[Card]) -> List[Card]:
        """Find the actual best 5-card hand from 7 cards"""
        best_rank = HandRank.HIGH_CARD
        best_tiebreakers = []
        best_five = []

        for combo in itertools.combinations(cards, 5):
            rank, tiebreakers = HandEvaluator._evaluate_5_cards(list(combo))
            if rank.value > best_rank.value or (rank == best_rank and tiebreakers > best_tiebreakers):
                best_rank = rank
                best_tiebreakers = tiebreakers
                best_five = list(combo)

        # Sort the best five cards for display (high to low)
        best_five.sort(key=lambda x: x.rank, reverse=True)
        return best_five

    @staticmethod
    def _rank_to_string(rank: int) -> str:
        """Convert rank number to readable string"""
        rank_map = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        return rank_map.get(rank, str(rank))


    @staticmethod
    def _create_readable_description(rank: HandRank, tiebreakers: List[int], best_five: List[Card]) -> str:
        """Create human-readable hand description"""
        rank_str = HandEvaluator._rank_to_string
    
        if rank == HandRank.STRAIGHT_FLUSH:
            high_card = rank_str(tiebreakers[0])
            return f"Straight Flush, {high_card} high"
    
        elif rank == HandRank.FOUR_OF_A_KIND:
            quad_rank = rank_str(tiebreakers[0])
            kicker = rank_str(tiebreakers[1])
            return f"Four {quad_rank}s, {kicker} kicker"
    
        elif rank == HandRank.FULL_HOUSE:
            trips = rank_str(tiebreakers[0])
            pair = rank_str(tiebreakers[1])
            return f"Full House, {trips}s full of {pair}s"
    
        elif rank == HandRank.FLUSH:
            cards_str = " ".join(rank_str(r) for r in tiebreakers)
            return f"Flush: {cards_str}"
    
        elif rank == HandRank.STRAIGHT:
            high_card = rank_str(tiebreakers[0])
            return f"Straight, {high_card} high"
    
        elif rank == HandRank.THREE_OF_A_KIND:
            trips = rank_str(tiebreakers[0])
            kickers = " ".join(rank_str(r) for r in tiebreakers[1:])
            return f"Three {trips}s, {kickers} kickers"
    
        elif rank == HandRank.TWO_PAIR:
            high_pair = rank_str(tiebreakers[0])
            low_pair = rank_str(tiebreakers[1])
            kicker = rank_str(tiebreakers[2])
            return f"Two Pair: {high_pair}s and {low_pair}s, {kicker} kicker"
    
        elif rank == HandRank.ONE_PAIR:
            pair_rank = rank_str(tiebreakers[0])
            kickers = " ".join(rank_str(r) for r in tiebreakers[1:])
            return f"Pair of {pair_rank}s, {kickers} kickers"
    
        else:  # HIGH_CARD
            cards_str = " ".join(rank_str(r) for r in tiebreakers)
            return f"High Card: {cards_str}"

