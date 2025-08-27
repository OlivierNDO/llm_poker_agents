"""
Unit tests for src/agents.py
"""
import unittest

from src.game import (
    ActionType,
    Action,
    Player,
    GameState,
    Hand,
    TableConfig,
    TableManager,
    StatsTracker,
)
from src.agents import Agent  # interface only


class AlwaysCheckAgent(Agent):
    def get_action(self, player, game_state, valid_actions, stats_tracker=None):
        if ActionType.CHECK in valid_actions:
            return Action(ActionType.CHECK)
        if ActionType.CALL in valid_actions:
            return Action(ActionType.CALL)
        return Action(valid_actions[0])


class AlwaysCallAgent(Agent):
    def get_action(self, player, game_state, valid_actions, stats_tracker=None):
        if ActionType.CALL in valid_actions:
            return Action(ActionType.CALL)
        if ActionType.CHECK in valid_actions:
            return Action(ActionType.CHECK)
        return Action(valid_actions[0])


class BetThenFoldAgent(Agent):
    def __init__(self):
        self.used = False

    def get_action(self, player, game_state, valid_actions, stats_tracker=None):
        if not self.used and ActionType.BET in valid_actions:
            self.used = True
            return Action(ActionType.BET, 4)
        if ActionType.FOLD in valid_actions:
            return Action(ActionType.FOLD)
        return Action(valid_actions[0])


class RaiseThenCallAgent(Agent):
    def __init__(self):
        self.used = False

    def get_action(self, player, game_state, valid_actions, stats_tracker=None):
        if not self.used and ActionType.RAISE in valid_actions:
            self.used = True
            call_amt = max(0, game_state.current_bet - player.current_bet)
            size = 3 if player.chips - call_amt >= 3 else max(1, player.chips - call_amt)
            return Action(ActionType.RAISE, size)
        if ActionType.CALL in valid_actions:
            return Action(ActionType.CALL)
        if ActionType.CHECK in valid_actions:
            return Action(ActionType.CHECK)
        return Action(valid_actions[0])


class TestPlayerBasics(unittest.TestCase):
    def test_all_in_sets_flag_and_contributes_fully(self):
        p = Player('A', 7)
        gs = GameState([p])
        new_bet, pot_add = p.apply_action(Action(ActionType.ALL_IN), gs)
        self.assertTrue(p.is_all_in)
        self.assertEqual(p.chips, 0)
        self.assertEqual(p.current_bet, 7)
        self.assertEqual(pot_add, 7)

    def test_get_valid_actions_when_no_current_bet(self):
        p = Player('A', 10)
        gs = GameState([p])
        acts = p.get_valid_actions(gs)
        self.assertIn(ActionType.CHECK, acts)
        self.assertIn(ActionType.BET, acts)

    def test_get_valid_actions_when_facing_bet(self):
        p = Player('A', 10)
        gs = GameState([p])
        gs.current_bet = 6
        p.current_bet = 0
        acts = p.get_valid_actions(gs)
        self.assertIn(ActionType.CALL, acts)
        self.assertIn(ActionType.RAISE, acts)
        self.assertIn(ActionType.ALL_IN, acts)


class TestGameStateCore(unittest.TestCase):
    def setUp(self):
        self.p1 = Player('A', 20)
        self.p2 = Player('B', 20)
        self.gs = GameState([self.p1, self.p2])

    def test_apply_action_updates_pot_and_history(self):
        self.gs.apply_action(self.p1, Action(ActionType.BET, 5))
        self.assertEqual(self.gs.pot, 5)
        self.assertEqual(self.gs.current_bet, 5)
        self.assertEqual(len(self.gs.action_history), 1)
        self.assertEqual(self.gs.player_pot_contributions['A'], 5)

    def test_is_betting_complete_trivial_single_active(self):
        self.p2.is_active = False
        self.assertTrue(self.gs.is_betting_complete())


class TestHandPhases(unittest.TestCase):
    def test_setup_and_hole_cards(self):
        players = [Player('A', 30), Player('B', 30)]
        h = Hand(players)
        msg1 = h.execute_next_step()
        self.assertIn('NEW HAND SETUP', msg1)
        self.assertEqual(h.phase, 'hole_cards')
        msg2 = h.execute_next_step()
        self.assertIn('dealt', msg2)
        self.assertEqual(h.phase, 'preflop')
        for p in players:
            self.assertIsNotNone(p.hole_cards)
            self.assertEqual(len(p.hole_cards), 2)

    def test_round_advances_when_all_have_acted(self):
        players = [Player('A', 50, AlwaysCheckAgent()), Player('B', 50, AlwaysCheckAgent())]
        tm = TableManager(players, TableConfig(small_blind=1, big_blind=2))
        h = Hand(players)
        h.execute_next_step()
        h.execute_next_step()
        tm._post_blinds(h)
        self.assertEqual(h.phase, 'preflop')
        for _ in range(10):
            out = h.execute_next_step()
            if 'Preflop betting complete' in out or 'Betting complete' in out:
                break
        self.assertIn(h.phase, ['flop', 'complete'])


class TestBlindsAndPositions(unittest.TestCase):
    def test_post_blinds_sets_current_bet_and_contributions(self):
        players = [Player('A', 10), Player('B', 10), Player('C', 10)]
        tm = TableManager(players, TableConfig(small_blind=1, big_blind=2))
        h = Hand(players)
        h.execute_next_step()
        h.execute_next_step()
        tm._post_blinds(h)
        gs = h.game_state
        self.assertEqual(gs.current_bet, tm.config.big_blind)
        self.assertEqual(sum(gs.player_pot_contributions.values()), tm.config.small_blind + tm.config.big_blind)
        self.assertEqual(len(gs.action_history), 2)


class TestIntegrationWithAgents(unittest.TestCase):
    def test_bet_then_fold_ends_hand_and_awards_pot(self):
        a = Player('Agg', 40, BetThenFoldAgent())
        c = Player('Call', 40, AlwaysCallAgent())
        players = [a, c]
        tm = TableManager(players, TableConfig(small_blind=1, big_blind=2))
        h = Hand(players)
        h.execute_next_step()
        h.execute_next_step()
        tm._post_blinds(h)
        while not h.is_complete:
            h.execute_next_step()
        total = sum(p.chips for p in players)
        self.assertEqual(total, 80)

    def test_raise_then_call_progresses_and_updates_pot(self):
        r = Player('Raiser', 50, RaiseThenCallAgent())
        c = Player('Caller', 50, AlwaysCallAgent())
        players = [r, c]
        tm = TableManager(players, TableConfig(small_blind=1, big_blind=2))
        h = Hand(players)
        h.execute_next_step()
        h.execute_next_step()
        tm._post_blinds(h)
        pre_pot = h.game_state.pot
        for _ in range(6):
            if h.is_complete:
                break
            h.execute_next_step()
        self.assertGreaterEqual(h.game_state.pot, pre_pot)


class TestStatsTrackerRecording(unittest.TestCase):
    def test_stats_update_on_hand_end(self):
        a = Player('A', 20, AlwaysCheckAgent())
        b = Player('B', 20, AlwaysCheckAgent())
        st = StatsTracker()
        h = Hand([a, b], stats_tracker=st)
        h.execute_next_step()
        h.execute_next_step()
        tm = TableManager([a, b], TableConfig(), stats_tracker=st)
        tm._post_blinds(h)
        while not h.is_complete:
            h.execute_next_step()
        self.assertGreaterEqual(st.get_player_stats('A').hands_played, 1)
        self.assertGreaterEqual(st.get_player_stats('B').hands_played, 1)


if __name__ == '__main__':
    unittest.main()
