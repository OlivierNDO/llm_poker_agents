"""
Unit tests for src/game.py
"""
import unittest
from src.game import (
    Deck, PlayerStats, ActionType, Action,
    Player, GameState, Hand, TableConfig, TableManager, StatsTracker
)

class TestDeck(unittest.TestCase):
    def test_deck_resets_to_52_unique_cards(self):
        deck = Deck()
        self.assertEqual(len(deck.cards), 52)
        unique_cards = {(c.rank, c.suit) for c in deck.cards}
        self.assertEqual(len(unique_cards), 52)

    def test_deal_reduces_deck_size(self):
        deck = Deck()
        first_card = deck.deal()
        self.assertEqual(len(deck.cards), 51)
        self.assertNotIn(first_card, deck.cards)


class TestPlayerStats(unittest.TestCase):
    def test_vpip_and_pfr_computation(self):
        stats = PlayerStats()
        stats.hands_played = 10
        stats.vpip_hands = 5
        stats.pfr_hands = 2
        self.assertEqual(stats.get_vpip_percent(), 50.0)
        self.assertEqual(stats.get_pfr_percent(), 20.0)

    def test_aggression_factor_infinite(self):
        stats = PlayerStats(total_bets=3, total_raises=2, total_calls=0)
        self.assertEqual(stats.get_aggression_factor(), float("inf"))

    def test_readable_stats_with_small_sample(self):
        stats = PlayerStats(hands_played=5, vpip_hands=2, pfr_hands=1)
        out = stats.get_readable_stats()
        self.assertIn("Warning", out)
        self.assertIn("VPIP", out)


class TestPlayer(unittest.TestCase):
    def setUp(self):
        self.player = Player("Alice", 100)

    def test_valid_actions_with_no_current_bet(self):
        gs = GameState([self.player])
        actions = self.player.get_valid_actions(gs)
        self.assertIn(ActionType.CHECK, actions)
        self.assertIn(ActionType.BET, actions)

    def test_call_and_raise_actions(self):
        gs = GameState([self.player])
        gs.current_bet = 10
        self.player.current_bet = 0
        actions = self.player.get_valid_actions(gs)
        self.assertIn(ActionType.CALL, actions)
        self.assertIn(ActionType.RAISE, actions)

    def test_all_in_sets_flag_and_contributes_fully(self):
        gs = GameState([self.player])
        action = Action(ActionType.ALL_IN)
        new_bet, pot_addition = self.player.apply_action(action, gs)
        self.assertTrue(self.player.is_all_in)
        self.assertEqual(pot_addition, 100)


class TestGameState(unittest.TestCase):
    def setUp(self):
        self.p1 = Player("Alice", 100)
        self.p2 = Player("Bob", 100)
        self.gs = GameState([self.p1, self.p2])

    def test_apply_action_records_history(self):
        action = Action(ActionType.BET, amount=20)
        self.gs.apply_action(self.p1, action)
        self.assertEqual(self.gs.pot, 20)
        self.assertEqual(self.gs.current_bet, 20)
        self.assertEqual(len(self.gs.action_history), 1)

    def test_is_betting_complete_with_single_player(self):
        self.p2.is_active = False
        self.assertTrue(self.gs.is_betting_complete())


class TestHand(unittest.TestCase):
    def setUp(self):
        self.players = [Player("Alice", 100), Player("Bob", 100)]
        self.hand = Hand(self.players)

    def test_hand_setup_resets_and_advances_phase(self):
        msg = self.hand._setup_hand()
        self.assertIn("NEW HAND SETUP", msg)
        self.assertEqual(self.hand.phase, "hole_cards")

    def test_deal_hole_cards_assigns_two_each(self):
        self.hand._setup_hand()
        msg = self.hand._deal_hole_cards()
        self.assertIn("dealt", msg)
        for p in self.players:
            self.assertIsNotNone(p.hole_cards)
            self.assertEqual(len(p.hole_cards), 2)


class TestTableManager(unittest.TestCase):
    def setUp(self):
        self.players = [Player("A", 50), Player("B", 50), Player("C", 50)]
        self.config = TableConfig(small_blind=1, big_blind=2)
        self.tm = TableManager(self.players, self.config)

    def test_count_funded_players(self):
        self.assertEqual(self.tm._count_funded_players(), 3)
        self.players[0].chips = 0
        self.assertEqual(self.tm._count_funded_players(), 2)

    def test_advance_button_rotates_dealer(self):
        old = self.tm.dealer_index
        self.tm._advance_button()
        self.assertNotEqual(self.tm.dealer_index, old)

    def test_post_blinds_sets_current_bet(self):
        hand = Hand(self.players)
        self.tm._post_blinds(hand)
        self.assertEqual(hand.game_state.current_bet, 2)
        sb_player = hand.players[1]  # depends on dealer
        self.assertGreaterEqual(sb_player.current_bet, 1)


if __name__ == "__main__":
    unittest.main()
