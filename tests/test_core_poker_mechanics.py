"""
Unit tests for src/core_poker_mechanics.py
"""
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import random
from collections import Counter
import itertools

# Import all classes from the original code
from src.core_poker_mechanics import (
    Suit, Card, GameState, HandRank, HandEvaluator, 
)


def rank_to_char(self):
    """Convert rank number to character"""
    rank_map = {
        14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
         9: '9',  8: '8',  7: '7',  6: '6',  5: '5',
         4: '4',  3: '3',  2: '2'
    }
    return rank_map.get(self.rank, str(self.rank))


class TestCard(unittest.TestCase):
    def test_card_creation(self):
        """Test card creation with valid rank and suit"""
        card = Card(14, Suit.SPADES)
        self.assertEqual(card.rank, 14)
        self.assertEqual(card.suit, Suit.SPADES)

    def test_card_string_representation(self):
        """Test card string representation for all ranks and suits"""
        # Test face cards
        self.assertEqual(str(Card(11, Suit.HEARTS)), "J♥")
        self.assertEqual(str(Card(12, Suit.DIAMONDS)), "Q♦")
        self.assertEqual(str(Card(13, Suit.CLUBS)), "K♣")
        self.assertEqual(str(Card(14, Suit.SPADES)), "A♠")

        # Test number cards
        self.assertEqual(str(Card(2, Suit.HEARTS)), "2♥")
        self.assertEqual(str(Card(10, Suit.DIAMONDS)), "10♦")

        # Test all suits
        self.assertEqual(str(Card(7, Suit.CLUBS)), "7♣")
        self.assertEqual(str(Card(7, Suit.DIAMONDS)), "7♦")
        self.assertEqual(str(Card(7, Suit.HEARTS)), "7♥")
        self.assertEqual(str(Card(7, Suit.SPADES)), "7♠")

    def test_card_equality(self):
        """Test card equality comparison"""
        card1 = Card(14, Suit.SPADES)
        card2 = Card(14, Suit.SPADES)
        card3 = Card(14, Suit.HEARTS)
        card4 = Card(13, Suit.SPADES)

        self.assertEqual(card1, card2)
        self.assertNotEqual(card1, card3)
        self.assertNotEqual(card1, card4)

    def test_card_hash(self):
        """Test card hashing for use in sets/dicts"""
        card1 = Card(14, Suit.SPADES)
        card2 = Card(14, Suit.SPADES)
        card3 = Card(14, Suit.HEARTS)

        self.assertEqual(hash(card1), hash(card2))
        self.assertNotEqual(hash(card1), hash(card3))

        # Test in set
        card_set = {card1, card2, card3}
        self.assertEqual(len(card_set), 2)  # card1 and card2 are same


class TestGameState(unittest.TestCase):
    def test_game_state_creation(self):
        """Test game state creation"""
        hands = [(Card(14, Suit.SPADES), Card(13, Suit.HEARTS))]
        board = [Card(12, Suit.DIAMONDS), Card(11, Suit.CLUBS)]
        state = GameState(hands, board, 4)

        self.assertEqual(len(state.player_hands), 1)
        self.assertEqual(len(state.board), 2)
        self.assertEqual(state.num_players, 4)

    def test_game_state_string_representation(self):
        """Test game state string representation"""
        hands = [(Card(14, Suit.SPADES), Card(13, Suit.HEARTS)), None]
        board = [Card(12, Suit.DIAMONDS)]
        state = GameState(hands, board, 2)

        str_repr = str(state)
        self.assertIn("Players: 2", str_repr)
        self.assertIn("Player 0: A♠ K♥", str_repr)
        self.assertIn("Player 1: [Hidden]", str_repr)
        self.assertIn("Board: Q♦", str_repr)

    def test_game_state_empty_board(self):
        """Test game state with empty board"""
        hands = [(Card(14, Suit.SPADES), Card(13, Suit.HEARTS))]
        state = GameState(hands, [], 2)

        str_repr = str(state)
        self.assertIn("Board: Empty", str_repr)


class TestHandEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up common test cards"""
        self.royal_flush_cards = [
            Card(14, Suit.SPADES), Card(13, Suit.SPADES), Card(12, Suit.SPADES),
            Card(11, Suit.SPADES), Card(10, Suit.SPADES)
        ]

        self.straight_flush_cards = [
            Card(9, Suit.HEARTS), Card(8, Suit.HEARTS), Card(7, Suit.HEARTS),
            Card(6, Suit.HEARTS), Card(5, Suit.HEARTS)
        ]

        self.four_of_a_kind_cards = [
            Card(14, Suit.SPADES), Card(14, Suit.HEARTS), Card(14, Suit.DIAMONDS),
            Card(14, Suit.CLUBS), Card(13, Suit.SPADES)
        ]

        self.full_house_cards = [
            Card(14, Suit.SPADES), Card(14, Suit.HEARTS), Card(14, Suit.DIAMONDS),
            Card(13, Suit.CLUBS), Card(13, Suit.SPADES)
        ]

        self.flush_cards = [
            Card(14, Suit.SPADES), Card(12, Suit.SPADES), Card(10, Suit.SPADES),
            Card(8, Suit.SPADES), Card(6, Suit.SPADES)
        ]

        self.straight_cards = [
            Card(14, Suit.SPADES), Card(13, Suit.HEARTS), Card(12, Suit.DIAMONDS),
            Card(11, Suit.CLUBS), Card(10, Suit.SPADES)
        ]

        self.ace_low_straight_cards = [
            Card(14, Suit.SPADES), Card(5, Suit.HEARTS), Card(4, Suit.DIAMONDS),
            Card(3, Suit.CLUBS), Card(2, Suit.SPADES)
        ]

        self.three_of_a_kind_cards = [
            Card(14, Suit.SPADES), Card(14, Suit.HEARTS), Card(14, Suit.DIAMONDS),
            Card(13, Suit.CLUBS), Card(12, Suit.SPADES)
        ]

        self.two_pair_cards = [
            Card(14, Suit.SPADES), Card(14, Suit.HEARTS), Card(13, Suit.DIAMONDS),
            Card(13, Suit.CLUBS), Card(12, Suit.SPADES)
        ]

        self.one_pair_cards = [
            Card(14, Suit.SPADES), Card(14, Suit.HEARTS), Card(13, Suit.DIAMONDS),
            Card(12, Suit.CLUBS), Card(11, Suit.SPADES)
        ]

        self.high_card_cards = [
            Card(14, Suit.SPADES), Card(13, Suit.HEARTS), Card(11, Suit.DIAMONDS),
            Card(9, Suit.CLUBS), Card(7, Suit.SPADES)
        ]

    def test_evaluate_5_cards_straight_flush(self):
        """Test straight flush evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.royal_flush_cards)
        self.assertEqual(rank, HandRank.STRAIGHT_FLUSH)
        self.assertEqual(tiebreakers, [14])

        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.straight_flush_cards)
        self.assertEqual(rank, HandRank.STRAIGHT_FLUSH)
        self.assertEqual(tiebreakers, [9])

    def test_evaluate_5_cards_four_of_a_kind(self):
        """Test four of a kind evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.four_of_a_kind_cards)
        self.assertEqual(rank, HandRank.FOUR_OF_A_KIND)
        self.assertEqual(tiebreakers, [14, 13])

    def test_evaluate_5_cards_full_house(self):
        """Test full house evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.full_house_cards)
        self.assertEqual(rank, HandRank.FULL_HOUSE)
        self.assertEqual(tiebreakers, [14, 13])

    def test_evaluate_5_cards_flush(self):
        """Test flush evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.flush_cards)
        self.assertEqual(rank, HandRank.FLUSH)
        self.assertEqual(tiebreakers, [14, 12, 10, 8, 6])

    def test_evaluate_5_cards_straight(self):
        """Test straight evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.straight_cards)
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [14])

    def test_evaluate_5_cards_ace_low_straight(self):
        """Test ace-low straight evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.ace_low_straight_cards)
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [5])  # 5 is high card in ace-low straight

    def test_evaluate_5_cards_three_of_a_kind(self):
        """Test three of a kind evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.three_of_a_kind_cards)
        self.assertEqual(rank, HandRank.THREE_OF_A_KIND)
        self.assertEqual(tiebreakers, [14, 13, 12])

    def test_evaluate_5_cards_two_pair(self):
        """Test two pair evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.two_pair_cards)
        self.assertEqual(rank, HandRank.TWO_PAIR)
        self.assertEqual(tiebreakers, [14, 13, 12])

    def test_evaluate_5_cards_one_pair(self):
        """Test one pair evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.one_pair_cards)
        self.assertEqual(rank, HandRank.ONE_PAIR)
        self.assertEqual(tiebreakers, [14, 13, 12, 11])

    def test_evaluate_5_cards_high_card(self):
        """Test high card evaluation"""
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(self.high_card_cards)
        self.assertEqual(rank, HandRank.HIGH_CARD)
        self.assertEqual(tiebreakers, [14, 13, 11, 9, 7])

    def test_evaluate_hand_7_cards(self):
        """Test 7-card hand evaluation"""
        # Create 7 cards that include a flush
        seven_cards = self.flush_cards + [Card(2, Suit.HEARTS), Card(3, Suit.DIAMONDS)]
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        self.assertEqual(rank, HandRank.FLUSH)
        self.assertEqual(tiebreakers, [14, 12, 10, 8, 6])

    def test_evaluate_hand_invalid_length(self):
        """Test hand evaluation with invalid card count"""
        with self.assertRaises(ValueError):
            HandEvaluator.evaluate_hand([Card(14, Suit.SPADES)])

    def test_find_best_five_cards(self):
        """Test finding best 5 cards from 7"""
        seven_cards = self.flush_cards + [Card(2, Suit.HEARTS), Card(3, Suit.DIAMONDS)]
        best_five = HandEvaluator._find_best_five_cards(seven_cards)
        self.assertEqual(len(best_five), 5)
        # Should be the flush cards, sorted high to low
        expected_ranks = [14, 12, 10, 8, 6]
        actual_ranks = [card.rank for card in best_five]
        self.assertEqual(actual_ranks, expected_ranks)

    def test_rank_to_string(self):
        """Test rank to string conversion"""
        self.assertEqual(HandEvaluator._rank_to_string(11), 'J')
        self.assertEqual(HandEvaluator._rank_to_string(12), 'Q')
        self.assertEqual(HandEvaluator._rank_to_string(13), 'K')
        self.assertEqual(HandEvaluator._rank_to_string(14), 'A')
        self.assertEqual(HandEvaluator._rank_to_string(7), '7')

    def test_create_readable_description(self):
        """Test readable hand descriptions"""
        # Test straight flush
        desc = HandEvaluator._create_readable_description(
            HandRank.STRAIGHT_FLUSH, [14], self.royal_flush_cards
        )
        self.assertEqual(desc, "Straight Flush, A high")

        # Test four of a kind
        desc = HandEvaluator._create_readable_description(
            HandRank.FOUR_OF_A_KIND, [14, 13], self.four_of_a_kind_cards
        )
        self.assertEqual(desc, "Four As, K kicker")

        # Test full house
        desc = HandEvaluator._create_readable_description(
            HandRank.FULL_HOUSE, [14, 13], self.full_house_cards
        )
        self.assertEqual(desc, "Full House, As full of Ks")

        # Test flush
        desc = HandEvaluator._create_readable_description(
            HandRank.FLUSH, [14, 12, 10, 8, 6], self.flush_cards
        )
        self.assertEqual(desc, "Flush: A Q 10 8 6")

        # Test straight
        desc = HandEvaluator._create_readable_description(
            HandRank.STRAIGHT, [14], self.straight_cards
        )
        self.assertEqual(desc, "Straight, A high")

        # Test three of a kind
        desc = HandEvaluator._create_readable_description(
            HandRank.THREE_OF_A_KIND, [14, 13, 12], self.three_of_a_kind_cards
        )
        self.assertEqual(desc, "Three As, K Q kickers")

        # Test two pair
        desc = HandEvaluator._create_readable_description(
            HandRank.TWO_PAIR, [14, 13, 12], self.two_pair_cards
        )
        self.assertEqual(desc, "Two Pair: As and Ks, Q kicker")

        # Test one pair
        desc = HandEvaluator._create_readable_description(
            HandRank.ONE_PAIR, [14, 13, 12, 11], self.one_pair_cards
        )
        self.assertEqual(desc, "Pair of As, K Q J kickers")

        # Test high card
        desc = HandEvaluator._create_readable_description(
            HandRank.HIGH_CARD, [14, 13, 11, 9, 7], self.high_card_cards
        )
        self.assertEqual(desc, "High Card: A K J 9 7")

    def test_hand_comparison_edge_cases(self):
        """Test edge cases in hand comparison"""
        # Test multiple kickers in four of a kind
        cards1 = [Card(14, Suit.SPADES), Card(14, Suit.HEARTS), Card(14, Suit.DIAMONDS),
                  Card(14, Suit.CLUBS), Card(13, Suit.SPADES)]
        cards2 = [Card(14, Suit.SPADES), Card(14, Suit.HEARTS), Card(14, Suit.DIAMONDS),
                  Card(14, Suit.CLUBS), Card(12, Suit.SPADES)]

        rank1, tie1 = HandEvaluator._evaluate_5_cards(cards1)
        rank2, tie2 = HandEvaluator._evaluate_5_cards(cards2)

        self.assertEqual(rank1, rank2)
        self.assertEqual(rank1, HandRank.FOUR_OF_A_KIND)
        self.assertGreater(tie1, tie2)  # King kicker beats Queen kicker



class TestPeerReviewIssues(unittest.TestCase):
    """Tests specifically targeting the peer review issues"""

    def test_straight_flush_false_positive_issue1(self):
        """
        Test for Issue #1: Straight flush false positive
        Cards that have a straight AND a flush but not the SAME cards forming both
        """
        # Test case: 4 spades (flush potential) + 1 heart that completes a straight
        # This should NOT be a straight flush
        cards = [
            Card(10, Suit.SPADES),  # Part of flush
            Card(11, Suit.SPADES),  # Part of flush  
            Card(12, Suit.SPADES),  # Part of flush
            Card(13, Suit.SPADES),  # Part of flush
            Card(14, Suit.HEARTS)   # Completes straight A-K-Q-J-10 but different suit
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        
        # This should be STRAIGHT, not STRAIGHT_FLUSH
        # Because while we have A-K-Q-J-10 straight, it's not all the same suit
        self.assertEqual(rank, HandRank.STRAIGHT, 
                        "Cards with 4 spades + 1 heart should be STRAIGHT, not STRAIGHT_FLUSH")
        self.assertEqual(tiebreakers, [14])

    def test_straight_flush_true_positive(self):
        """Test that actual straight flushes are detected correctly"""
        # True straight flush - all same suit and consecutive
        cards = [
            Card(10, Suit.SPADES),
            Card(11, Suit.SPADES), 
            Card(12, Suit.SPADES),
            Card(13, Suit.SPADES),
            Card(14, Suit.SPADES)
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.STRAIGHT_FLUSH)
        self.assertEqual(tiebreakers, [14])

    def test_flush_with_exactly_5_cards_issue2(self):
        """
        Test for Issue #2: Flush logic for exactly 5 cards
        Since _evaluate_5_cards only handles exactly 5 cards, 
        the current logic should be correct for this case
        """
        # All 5 cards same suit - should be flush
        cards = [
            Card(14, Suit.HEARTS),
            Card(12, Suit.HEARTS),
            Card(10, Suit.HEARTS),
            Card(8, Suit.HEARTS),
            Card(6, Suit.HEARTS)
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.FLUSH)
        self.assertEqual(tiebreakers, [14, 12, 10, 8, 6])

    def test_straight_detection_with_exactly_5_cards_issue3(self):
        """
        Test for Issue #3: Straight detection with exactly 5 cards
        Since _evaluate_5_cards only handles exactly 5 cards,
        the current logic should work for this case
        """
        # Exactly 5 cards forming a straight
        cards = [
            Card(10, Suit.SPADES),
            Card(11, Suit.HEARTS),
            Card(12, Suit.DIAMONDS),
            Card(13, Suit.CLUBS),
            Card(14, Suit.SPADES)
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [14])

    def test_mixed_flush_and_straight_cards(self):
        """
        Test edge case: Cards that could form flush OR straight but not both
        """
        # 3 hearts + 2 other suits, where all 5 cards happen to be consecutive
        cards = [
            Card(10, Suit.HEARTS),
            Card(11, Suit.HEARTS), 
            Card(12, Suit.HEARTS),
            Card(13, Suit.DIAMONDS),  # Breaks flush
            Card(14, Suit.CLUBS)      # Breaks flush
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        # Should be straight (10-J-Q-K-A), not flush
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [14])

    def test_ace_low_straight_flush(self):
        """Test ace-low straight flush (A-2-3-4-5 all same suit)"""
        cards = [
            Card(14, Suit.CLUBS),  # Ace
            Card(2, Suit.CLUBS),
            Card(3, Suit.CLUBS),
            Card(4, Suit.CLUBS),
            Card(5, Suit.CLUBS)
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.STRAIGHT_FLUSH)
        self.assertEqual(tiebreakers, [5])  # 5 high in ace-low

    def test_broken_straight_with_flush(self):
        """Test cards that are all same suit but don't form a straight"""
        cards = [
            Card(14, Suit.SPADES),
            Card(12, Suit.SPADES),
            Card(10, Suit.SPADES),
            Card(8, Suit.SPADES),
            Card(6, Suit.SPADES)
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        # Should be flush, not straight (A-Q-10-8-6 is not consecutive)
        self.assertEqual(rank, HandRank.FLUSH)
        self.assertEqual(tiebreakers, [14, 12, 10, 8, 6])

    def test_7_card_hand_with_potential_issues(self):
        """
        Test 7-card hand evaluation where different 5-card combinations
        could have different flush/straight properties
        """
        # 7 cards: 5 hearts (making flush) + 2 other cards that could make straight
        seven_cards = [
            Card(14, Suit.HEARTS),  # Part of flush
            Card(13, Suit.HEARTS),  # Part of flush
            Card(12, Suit.HEARTS),  # Part of flush
            Card(11, Suit.HEARTS),  # Part of flush
            Card(10, Suit.HEARTS),  # Part of flush - this makes straight flush!
            Card(9, Suit.DIAMONDS), # Extra card
            Card(8, Suit.CLUBS)     # Extra card
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        # Should find the straight flush A-K-Q-J-10 of hearts
        self.assertEqual(rank, HandRank.STRAIGHT_FLUSH)
        self.assertEqual(tiebreakers, [14])

    def test_7_card_hand_flush_vs_straight_choice(self):
        """
        Test 7-card hand where we could choose flush OR straight but not both
        """
        seven_cards = [
            # 5 clubs for flush
            Card(14, Suit.CLUBS),   
            Card(12, Suit.CLUBS),
            Card(10, Suit.CLUBS),
            Card(8, Suit.CLUBS),
            Card(6, Suit.CLUBS),
            # 2 cards that could make a different straight
            Card(13, Suit.HEARTS),  # Could make A-K-Q-J-10 straight with different combo
            Card(11, Suit.DIAMONDS)
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        # Should choose flush (higher rank than straight)
        self.assertEqual(rank, HandRank.FLUSH)
        # Should use the 5 clubs
        self.assertEqual(tiebreakers, [14, 12, 10, 8, 6])

    def test_edge_case_multiple_straights_in_7_cards(self):
        """
        Test case where 7 cards contain multiple possible straights
        """
        seven_cards = [
            Card(6, Suit.HEARTS),
            Card(7, Suit.CLUBS), 
            Card(8, Suit.DIAMONDS),
            Card(9, Suit.SPADES),
            Card(10, Suit.HEARTS),  # 6-7-8-9-10 straight
            Card(11, Suit.CLUBS),   # 7-8-9-10-J straight  
            Card(12, Suit.DIAMONDS) # 8-9-10-J-Q straight
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        # Should find best straight (8-9-10-J-Q, Q high)
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [12])  # Queen high

    def test_verify_issue_1_cannot_occur_with_5_cards(self):
        """
        Verify that Issue #1 cannot actually occur with exactly 5 cards.
        With exactly 5 cards, is_flush and is_straight apply to the SAME cards.
        """
        # This test demonstrates why Issue #1 doesn't apply to _evaluate_5_cards
        
        # Case 1: 4 spades + 1 heart forming straight (should be STRAIGHT)
        cards_mixed_straight = [
            Card(9, Suit.SPADES),   
            Card(10, Suit.SPADES),  
            Card(11, Suit.SPADES),  
            Card(12, Suit.SPADES), 
            Card(13, Suit.HEARTS)   # Different suit breaks flush
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards_mixed_straight)
        self.assertEqual(rank, HandRank.STRAIGHT)  # Not straight flush
        self.assertEqual(tiebreakers, [13])
        
        # Case 2: True straight flush (should be STRAIGHT_FLUSH)
        cards_true_sf = [
            Card(9, Suit.SPADES),   
            Card(10, Suit.SPADES),  
            Card(11, Suit.SPADES),  
            Card(12, Suit.SPADES), 
            Card(13, Suit.SPADES)   # Same suit = true straight flush
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards_true_sf)
        self.assertEqual(rank, HandRank.STRAIGHT_FLUSH)  # Correctly identified
        self.assertEqual(tiebreakers, [13])
        
        # The key insight: With exactly 5 cards, there's no way to have
        # "some cards form flush, different cards form straight"
        # because we only have 5 cards total!

    def test_could_issue_1_affect_7_card_evaluation(self):
        """
        Test if Issue #1 could theoretically affect the 7-card evaluation
        by creating a scenario where bad 5-card logic might pick wrong combination
        """
        # Create 7 cards where a buggy algorithm might pick the wrong 5
        seven_cards = [
            # 4 spades that almost make a flush
            Card(10, Suit.SPADES),
            Card(11, Suit.SPADES), 
            Card(12, Suit.SPADES),
            Card(13, Suit.SPADES),
            # 3 other cards that could complete straights
            Card(14, Suit.HEARTS),  # Completes 10-J-Q-K-A straight
            Card(9, Suit.DIAMONDS), # Completes 9-10-J-Q-K straight  
            Card(2, Suit.CLUBS)     # Random low card
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        
        # The correct evaluation should find the best 5-card hand
        # Could be: 10♠-J♠-Q♠-K♠-A♥ (straight, A high)
        # Or: 9♦-10♠-J♠-Q♠-K♠ (straight, K high)  
        # Should NOT be straight flush since no 5 cards are same suit
        
        self.assertEqual(rank, HandRank.STRAIGHT)
        # Should pick the ace-high straight
        self.assertEqual(tiebreakers, [14])



class TestPokerEdgeCases(unittest.TestCase):
    """Comprehensive edge case testing for poker hand evaluation"""

    def test_ace_wraparound_invalid_straight(self):
        """Test that K-A-2-3-4 is NOT a valid straight (ace doesn't wrap)"""
        cards = [
            Card(13, Suit.SPADES),  # King
            Card(14, Suit.HEARTS),  # Ace  
            Card(2, Suit.DIAMONDS), # Two
            Card(3, Suit.CLUBS),    # Three
            Card(4, Suit.SPADES)    # Four
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        # Should be high card (ace high), not straight
        self.assertEqual(rank, HandRank.HIGH_CARD)
        self.assertEqual(tiebreakers, [14, 13, 4, 3, 2])

    def test_ace_wraparound_invalid_straight_flush(self):
        """Test that K-A-2-3-4 of same suit is flush, not straight flush"""
        cards = [
            Card(13, Suit.SPADES),  # King
            Card(14, Suit.SPADES),  # Ace  
            Card(2, Suit.SPADES),   # Two
            Card(3, Suit.SPADES),   # Three
            Card(4, Suit.SPADES)    # Four
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        # Should be flush, not straight flush (no wraparound)
        self.assertEqual(rank, HandRank.FLUSH)
        self.assertEqual(tiebreakers, [14, 13, 4, 3, 2])

    def test_multiple_aces_in_7_cards(self):
        """Test 7-card hand with multiple aces"""
        seven_cards = [
            Card(14, Suit.SPADES),  # Ace of spades
            Card(14, Suit.HEARTS),  # Ace of hearts
            Card(14, Suit.DIAMONDS), # Ace of diamonds
            Card(14, Suit.CLUBS),   # Ace of clubs (four aces!)
            Card(13, Suit.SPADES),  # King
            Card(12, Suit.HEARTS),  # Queen
            Card(11, Suit.DIAMONDS) # Jack
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        self.assertEqual(rank, HandRank.FOUR_OF_A_KIND)
        self.assertEqual(tiebreakers, [14, 13])  # Four aces, king kicker

    def test_close_tiebreaker_scenarios(self):
        """Test hands that differ only by small tiebreakers"""
        # Two pair: Aces and Kings with Queen kicker
        cards1 = [Card(14, Suit.SPADES), Card(14, Suit.HEARTS), 
                  Card(13, Suit.DIAMONDS), Card(13, Suit.CLUBS), Card(12, Suit.SPADES)]
        
        # Two pair: Aces and Kings with Jack kicker  
        cards2 = [Card(14, Suit.SPADES), Card(14, Suit.HEARTS),
                  Card(13, Suit.DIAMONDS), Card(13, Suit.CLUBS), Card(11, Suit.SPADES)]
        
        rank1, tie1 = HandEvaluator._evaluate_5_cards(cards1)
        rank2, tie2 = HandEvaluator._evaluate_5_cards(cards2)
        
        self.assertEqual(rank1, rank2)  # Same rank
        self.assertEqual(rank1, HandRank.TWO_PAIR)
        self.assertGreater(tie1, tie2)  # Queen kicker beats Jack

    def test_straight_with_duplicate_ranks_in_7_cards(self):
        """Test 7-card hand with duplicate ranks but still contains straight"""
        seven_cards = [
            Card(10, Suit.SPADES),
            Card(10, Suit.HEARTS),  # Duplicate 10
            Card(11, Suit.DIAMONDS),
            Card(12, Suit.CLUBS),
            Card(13, Suit.SPADES), 
            Card(14, Suit.HEARTS),  # 10-J-Q-K-A straight present
            Card(2, Suit.DIAMONDS)  # Low card
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        # Should find the straight, not just the pair
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [14])

    def test_flush_tiebreaker_ordering(self):
        """Test that flush tiebreakers are ordered correctly (high to low)"""
        cards = [
            Card(14, Suit.HEARTS),  # Ace
            Card(9, Suit.HEARTS),   # Nine
            Card(7, Suit.HEARTS),   # Seven
            Card(5, Suit.HEARTS),   # Five
            Card(3, Suit.HEARTS)    # Three
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.FLUSH)
        # Should be ordered high to low
        self.assertEqual(tiebreakers, [14, 9, 7, 5, 3])

    def test_full_house_vs_flush_priority(self):
        """Test that full house beats flush when both are possible in 7 cards"""
        seven_cards = [
            Card(14, Suit.HEARTS),  # Ace of hearts
            Card(14, Suit.SPADES),  # Ace of spades
            Card(14, Suit.DIAMONDS), # Ace of diamonds (trip aces)
            Card(13, Suit.HEARTS),  # King of hearts
            Card(13, Suit.CLUBS),   # King of clubs (pair kings)
            Card(12, Suit.HEARTS),  # Queen of hearts
            Card(11, Suit.HEARTS)   # Jack of hearts (5 hearts for flush)
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        # Should choose full house (aces full of kings) over flush
        self.assertEqual(rank, HandRank.FULL_HOUSE)
        self.assertEqual(tiebreakers, [14, 13])

    def test_low_straight_vs_high_cards(self):
        """Test that low straight beats high cards"""
        cards = [
            Card(14, Suit.SPADES),  # Ace (low in this straight)
            Card(2, Suit.HEARTS),   # Two
            Card(3, Suit.DIAMONDS), # Three
            Card(4, Suit.CLUBS),    # Four
            Card(5, Suit.SPADES)    # Five (wheel straight)
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [5])  # 5-high straight

    def test_three_of_a_kind_kicker_evaluation(self):
        """Test three of a kind with proper kicker ordering"""
        cards = [
            Card(8, Suit.SPADES),   # Trip eights
            Card(8, Suit.HEARTS),
            Card(8, Suit.DIAMONDS),
            Card(14, Suit.CLUBS),   # Ace kicker
            Card(13, Suit.SPADES)   # King kicker
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.THREE_OF_A_KIND)
        # Should be [trip_rank, high_kicker, low_kicker]
        self.assertEqual(tiebreakers, [8, 14, 13])

    def test_one_pair_with_many_kickers(self):
        """Test one pair with multiple kickers in correct order"""
        cards = [
            Card(7, Suit.SPADES),   # Pair of sevens
            Card(7, Suit.HEARTS),
            Card(14, Suit.DIAMONDS), # Ace kicker
            Card(13, Suit.CLUBS),   # King kicker  
            Card(12, Suit.SPADES)   # Queen kicker
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.ONE_PAIR)
        # Should be [pair_rank, ace, king, queen]
        self.assertEqual(tiebreakers, [7, 14, 13, 12])

    def test_royal_flush_detection(self):
        """Test royal flush (A-K-Q-J-10 suited) is detected as straight flush"""
        cards = [
            Card(14, Suit.SPADES),  # Ace
            Card(13, Suit.SPADES),  # King
            Card(12, Suit.SPADES),  # Queen
            Card(11, Suit.SPADES),  # Jack
            Card(10, Suit.SPADES)   # Ten
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.STRAIGHT_FLUSH)
        self.assertEqual(tiebreakers, [14])  # Ace high

    def test_7_cards_complex_full_house(self):
        """Test complex 7-card scenario with multiple pairs and trips"""
        # Test: 4 of a kind + 3 of another kind
        seven_cards = [
            Card(14, Suit.SPADES), Card(14, Suit.HEARTS), 
            Card(14, Suit.DIAMONDS), Card(14, Suit.CLUBS),  # Four aces
            Card(13, Suit.SPADES), Card(13, Suit.HEARTS), 
            Card(13, Suit.DIAMONDS)  # Three kings
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        # Should pick four aces (higher than full house)
        self.assertEqual(rank, HandRank.FOUR_OF_A_KIND)
        self.assertEqual(tiebreakers, [14, 13])

    def test_trips_and_two_pair_makes_full_house(self):
        """Test 7-card evaluation with various challenging combinations"""
        # Two pair + three of a kind = full house
        seven_cards = [
            Card(10, Suit.SPADES), Card(10, Suit.HEARTS), Card(10, Suit.DIAMONDS),  # Trip tens
            Card(5, Suit.CLUBS), Card(5, Suit.SPADES),  # Pair fives
            Card(14, Suit.HEARTS), Card(2, Suit.DIAMONDS)  # Ace and deuce
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        self.assertEqual(rank, HandRank.FULL_HOUSE)
        self.assertEqual(tiebreakers, [10, 5])  # Tens full of fives

    def test_comparison_edge_case_equal_hands(self):
        """Test identical hands for proper equality"""
        cards1 = [Card(14, Suit.SPADES), Card(13, Suit.HEARTS), 
                  Card(12, Suit.DIAMONDS), Card(11, Suit.CLUBS), Card(10, Suit.SPADES)]
        
        cards2 = [Card(14, Suit.HEARTS), Card(13, Suit.SPADES), 
                  Card(12, Suit.CLUBS), Card(11, Suit.DIAMONDS), Card(10, Suit.HEARTS)]
        
        rank1, tie1 = HandEvaluator._evaluate_5_cards(cards1)
        rank2, tie2 = HandEvaluator._evaluate_5_cards(cards2)
        
        self.assertEqual(rank1, rank2)
        self.assertEqual(tie1, tie2)  # Should be identical tiebreakers

    def test_near_straight_flush_scenarios(self):
        """Test scenarios that are almost straight flushes"""
        # Four to straight flush + one off-suit
        cards = [
            Card(9, Suit.HEARTS),
            Card(10, Suit.HEARTS), 
            Card(11, Suit.HEARTS),
            Card(12, Suit.HEARTS),  # Four hearts in sequence
            Card(13, Suit.SPADES)   # King of spades (breaks flush)
        ]
        
        rank, tiebreakers = HandEvaluator._evaluate_5_cards(cards)
        self.assertEqual(rank, HandRank.STRAIGHT)  # Not straight flush
        self.assertEqual(tiebreakers, [13])

    def test_multiple_possible_straights_7_cards(self):
        """Test 7-card hand with multiple overlapping straights"""
        seven_cards = [
            Card(5, Suit.HEARTS),
            Card(6, Suit.CLUBS),
            Card(7, Suit.DIAMONDS),
            Card(8, Suit.SPADES),
            Card(9, Suit.HEARTS),
            Card(10, Suit.CLUBS),   # Multiple straights: 5-9, 6-10
            Card(2, Suit.DIAMONDS)  # Unrelated card
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [10])  # Should pick highest straight (6-10)

    def test_wheel_straight_in_7_cards(self):
        """Test A-2-3-4-5 straight in 7-card hand with higher cards"""
        seven_cards = [
            Card(14, Suit.SPADES),  # Ace
            Card(2, Suit.HEARTS),   # Two
            Card(3, Suit.DIAMONDS), # Three
            Card(4, Suit.CLUBS),    # Four
            Card(5, Suit.SPADES),   # Five (wheel)
            Card(13, Suit.HEARTS),  # King
            Card(12, Suit.DIAMONDS) # Queen
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        # Could make wheel (A-2-3-4-5) or high card (A-K-Q-5-4)
        # Straight should win
        self.assertEqual(rank, HandRank.STRAIGHT)
        self.assertEqual(tiebreakers, [5])  # Wheel straight (5 high)

    def test_backdoor_flush_in_7_cards(self):
        """Test 7-card hand where flush is possible but not obvious"""
        seven_cards = [
            Card(14, Suit.CLUBS),   # Ace of clubs
            Card(9, Suit.CLUBS),    # Nine of clubs
            Card(7, Suit.CLUBS),    # Seven of clubs
            Card(5, Suit.CLUBS),    # Five of clubs
            Card(3, Suit.CLUBS),    # Three of clubs (5 clubs = flush)
            Card(13, Suit.SPADES),  # King of spades
            Card(12, Suit.HEARTS)   # Queen of hearts
        ]
        
        rank, tiebreakers = HandEvaluator.evaluate_hand(seven_cards)
        self.assertEqual(rank, HandRank.FLUSH)
        self.assertEqual(tiebreakers, [14, 9, 7, 5, 3])





if __name__ == '__main__':
    unittest.main(verbosity=2)