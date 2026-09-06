"""Unit tests for Temporal-Difference Q-learning in QLearningCore (Stage 17D)."""

import unittest
from unittest.mock import patch

from brain.q_learning import QLearningCore
from brain.transition import Transition


class TestQLearningTemporalDifference(unittest.TestCase):
    def test_basic_td_update_formula_verification(self) -> None:
        """Specification test:

        old_q = 10.0
        reward = 5.0
        gamma = 0.85
        max_next_q = 20.0
        alpha = 0.1

        Target: 5.0 + 0.85 * 20.0 = 22.0
        Update: 10.0 + 0.1 * (22.0 - 10.0) = 11.2
        """
        ql = QLearningCore(learning_rate=0.10, discount_factor=0.85)

        # Setup source state Q(s, a) = 10.0
        ql.set_q("combat", "STATE_CURRENT", "DIVE_ALL_IN", 10.0)

        # Setup next state with max Q(s', a') = 20.0
        ql.init_state(
            "combat",
            "STATE_NEXT",
            {"KITE_AND_POKE": 5.0, "SWEET_SPOT_BURST": 20.0, "DIVE_ALL_IN": 10.0},
        )

        trans = Transition(
            state_key="STATE_CURRENT",
            action="DIVE_ALL_IN",
            reward=5.0,
            next_state_key="STATE_NEXT",
            done=False,
        )

        new_q = ql.update_td(trans, alpha=0.10)
        self.assertAlmostEqual(new_q, 11.2)
        self.assertAlmostEqual(ql.get_q("combat", "STATE_CURRENT", "DIVE_ALL_IN"), 11.2)

    def test_terminal_update_ignores_gamma(self) -> None:
        """Terminal transition (done=True):

        old_q = 10.0
        reward = 5.0
        done = True

        Target: 5.0 (no gamma * max_q contribution)
        Update: 10.0 + 0.1 * (5.0 - 10.0) = 9.5
        """
        ql = QLearningCore(learning_rate=0.10, discount_factor=0.85)
        ql.set_q("combat", "CRITICAL_STATE", "TACTICAL_RETREAT", 10.0)

        trans = Transition(
            state_key="CRITICAL_STATE",
            action="TACTICAL_RETREAT",
            reward=5.0,
            next_state_key=None,
            done=True,
        )

        new_q = ql.update_td(trans, alpha=0.10)
        self.assertAlmostEqual(new_q, 9.5)
        self.assertAlmostEqual(ql.get_q("combat", "CRITICAL_STATE", "TACTICAL_RETREAT"), 9.5)

    def test_non_terminal_with_none_next_state_raises_value_error(self) -> None:
        """done=False with next_state_key=None must raise ValueError."""
        ql = QLearningCore()
        # Create invalid transition bypassing normal post-init if possible or mock
        from unittest.mock import MagicMock

        invalid_trans = MagicMock(spec=Transition)
        invalid_trans.done = False
        invalid_trans.next_state_key = None

        with self.assertRaises(ValueError):
            ql.update_td(invalid_trans)

    def test_negative_reward_td_update(self) -> None:
        """Verifies TD update with negative reward:

        old_q = 15.0, reward = -10.0, gamma = 0.80, max_next_q = 20.0, alpha = 0.20
        target = -10.0 + 0.80 * 20.0 = 6.0
        new_q = 15.0 + 0.20 * (6.0 - 15.0) = 15.0 - 1.8 = 13.2
        """
        ql = QLearningCore(learning_rate=0.20, discount_factor=0.80)
        ql.set_q("farm", "FARM_S1", "FARM_APPROACH", 15.0)
        ql.set_q("farm", "FARM_S2", "FARM_SWEET_SPOT", 20.0)

        trans = Transition(
            state_key="FARM_S1",
            action="FARM_APPROACH",
            reward=-10.0,
            next_state_key="FARM_S2",
            done=False,
        )

        new_q = ql.update_td(trans)
        self.assertAlmostEqual(new_q, 13.2)

    def test_positive_future_value_increases_current_q(self) -> None:
        """High Q value in next_state actively pulls up current Q value."""
        ql = QLearningCore(learning_rate=0.50, discount_factor=0.85)
        ql.set_q("roam", "ROAM_A", "LANE_ADVANCE", 0.0)
        ql.set_q("roam", "ROAM_B", "CREEP_INTERCEPT", 50.0)

        trans = Transition("ROAM_A", "LANE_ADVANCE", 0.0, "ROAM_B", False)
        # target = 0.0 + 0.85 * 50.0 = 42.5
        # new_q = 0.0 + 0.50 * 42.5 = 21.25
        new_q = ql.update_td(trans)
        self.assertGreater(new_q, 20.0)

    def test_gamma_effect_comparison(self) -> None:
        """Different gamma values produce predictably distinct updates on the same transition."""
        ql_no_gamma = QLearningCore(learning_rate=0.25, discount_factor=0.0)
        ql_with_gamma = QLearningCore(learning_rate=0.25, discount_factor=0.85)

        ql_no_gamma.set_q("combat", "S1", "A1", 10.0)
        ql_no_gamma.set_q("combat", "S2", "A2", 40.0)

        ql_with_gamma.set_q("combat", "S1", "A1", 10.0)
        ql_with_gamma.set_q("combat", "S2", "A2", 40.0)

        trans = Transition("S1", "A1", 10.0, "S2", False)

        q_no_gamma = ql_no_gamma.update_td(trans)
        q_with_gamma = ql_with_gamma.update_td(trans)

        # With gamma=0: target = 10.0 + 0 = 10.0 -> update = 10.0 + 0.25*(10-10) = 10.0
        self.assertAlmostEqual(q_no_gamma, 10.0)

        # With gamma=0.85: target = 10.0 + 0.85*40 = 44.0 -> update = 10.0 + 0.25*(44-10) = 18.5
        self.assertAlmostEqual(q_with_gamma, 18.5)
        self.assertGreater(q_with_gamma, q_no_gamma)

    def test_alpha_effect_comparison(self) -> None:
        """Different alpha parameters control update speed toward target."""
        ql = QLearningCore(learning_rate=0.25, discount_factor=0.85)
        ql.set_q("farm", "F1", "A1", 0.0)
        ql.set_q("farm", "F2", "A2", 20.0)

        trans = Transition("F1", "A1", 10.0, "F2", False)
        # target = 10 + 0.85*20 = 27.0
        # with alpha=0.10: new_q = 0.0 + 0.10 * 27.0 = 2.7
        # with alpha=0.50: new_q = 0.0 + 0.50 * 27.0 = 13.5
        q_slow = ql.update_td(trans, alpha=0.10)
        self.assertAlmostEqual(q_slow, 2.7)

        ql.set_q("farm", "F1", "A1", 0.0)
        q_fast = ql.update_td(trans, alpha=0.50)
        self.assertAlmostEqual(q_fast, 13.5)

    def test_no_random_or_exploration_called_during_update_td(self) -> None:
        """update_td must strictly be deterministic and never call random.random or choice."""
        ql = QLearningCore()
        ql.set_q("combat", "S1", "A1", 10.0)
        trans = Transition("S1", "A1", 5.0, None, True)

        with patch("random.random", side_effect=RuntimeError("random.random called!")):
            with patch("random.choice", side_effect=RuntimeError("random.choice called!")):
                new_q = ql.update_td(trans)
                self.assertIsNotNone(new_q)

    def test_v1_update_q_formula_remains_strictly_unchanged(self) -> None:
        """update_q() must continue to use V1 formula Q + alpha*(R - Q) without gamma."""
        ql = QLearningCore(learning_rate=0.25, discount_factor=0.85)
        ql.set_q("combat", "COMBAT_STATE", "KITE_AND_POKE", 10.0)

        # V1 update: old=10.0, reward=20.0, alpha=0.25 -> 10 + 0.25 * (20 - 10) = 12.5
        v1_result = ql.update_q("combat", "COMBAT_STATE", "KITE_AND_POKE", reward=20.0)
        self.assertAlmostEqual(v1_result, 12.5)


if __name__ == "__main__":
    unittest.main()
