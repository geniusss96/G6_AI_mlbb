"""
Unit tests for brain.q_learning (QLearningCore).
Verifies exact V1 mathematical properties, formula regression, default state creation,
positive/negative reward updates, gamma/epsilon preservation, and architecture isolation.
"""

import os
import sys
import unittest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from brain.q_learning import QLearningCore


class TestQLearningCore(unittest.TestCase):
    def setUp(self):
        self.ql = QLearningCore(learning_rate=0.25, discount_factor=0.85, exploration_rate=0.20)

    def test_missing_state_returns_default(self):
        val = self.ql.get_q("combat", "NON_EXISTENT_STATE", "DIVE_ALL_IN", default=0.0)
        self.assertEqual(val, 0.0)

    def test_ensure_state_creates_defaults(self):
        defaults = {"ACTION_A": 15.0, "ACTION_B": -5.0}
        created = self.ql.ensure_state("farm", "TEST_FARM_STATE", defaults)
        self.assertEqual(created, defaults)
        self.assertEqual(self.ql.get_q("farm", "TEST_FARM_STATE", "ACTION_A"), 15.0)
        self.assertEqual(self.ql.get_q("farm", "TEST_FARM_STATE", "ACTION_B"), -5.0)

        # Calling again should NOT overwrite modified values
        self.ql.set_q("farm", "TEST_FARM_STATE", "ACTION_A", 25.0)
        self.ql.ensure_state("farm", "TEST_FARM_STATE", defaults)
        self.assertEqual(self.ql.get_q("farm", "TEST_FARM_STATE", "ACTION_A"), 25.0)

    def test_set_and_get_q(self):
        self.ql.set_q("roam", "SCOUT_HOT_ZONE", "LANE_ADVANCE", 42.126)
        # Should be rounded to 2 decimal places per V1
        self.assertEqual(self.ql.get_q("roam", "SCOUT_HOT_ZONE", "LANE_ADVANCE"), 42.13)

    def test_formula_regression(self):
        """
        Critical regression requirement:
        old_q = 10, alpha = 0.1, reward = 20
        new_q = old_q + alpha * (reward - old_q) = 10 + 0.1 * 10 = 11.0
        Must NOT use classical Q-learning formula with gamma!
        """
        self.ql.set_q("combat", "TEST_STATE", "KITE_AND_POKE", 10.0)
        new_q = self.ql.update_q("combat", "TEST_STATE", "KITE_AND_POKE", reward=20.0, alpha=0.1)
        self.assertAlmostEqual(new_q, 11.0)
        self.assertAlmostEqual(self.ql.get_q("combat", "TEST_STATE", "KITE_AND_POKE"), 11.0)

    def test_positive_reward_update_with_default_alpha(self):
        """
        V1 kill reward: old_q = 0.0, alpha = 0.25, reward = 60.0
        new_q = 0.0 + 0.25 * (60.0 - 0.0) = 15.0
        """
        self.ql.set_q("combat", "ENEMY_FOUND", "DIVE_ALL_IN", 0.0)
        new_q = self.ql.update_q("combat", "ENEMY_FOUND", "DIVE_ALL_IN", reward=60.0)
        self.assertAlmostEqual(new_q, 15.0)

    def test_negative_reward_update_with_default_alpha(self):
        """
        V1 death penalty: old_q = 10.0, alpha = 0.25, reward = -40.0
        new_q = 10.0 + 0.25 * (-40.0 - 10.0) = 10.0 - 12.5 = -2.5
        """
        self.ql.set_q("combat", "COMBAT_STATE", "SWEET_SPOT_BURST", 10.0)
        new_q = self.ql.update_q("combat", "COMBAT_STATE", "SWEET_SPOT_BURST", reward=-40.0)
        self.assertAlmostEqual(new_q, -2.5)

    def test_repeated_updates_convergence(self):
        self.ql.set_q("farm", "FARM_STATE", "FARM_APPROACH", 0.0)
        for _ in range(20):
            self.ql.update_q("farm", "FARM_STATE", "FARM_APPROACH", reward=30.0)
        # Approaches 30.0 asymptotically
        self.assertGreater(self.ql.get_q("farm", "FARM_STATE", "FARM_APPROACH"), 29.0)

    def test_gamma_and_epsilon_behavior_preserved(self):
        """Gamma and domain epsilon values must match V1 defaults."""
        self.assertEqual(self.ql.alpha, 0.25)
        self.assertEqual(self.ql.gamma, 0.85)
        self.assertEqual(self.ql.epsilon, 0.20)
        self.assertEqual(self.ql.roam_epsilon, 0.20)
        self.assertEqual(self.ql.farm_epsilon, 0.10)
        self.assertEqual(self.ql.combat_epsilon, 0.08)

    def test_get_best_action_selection(self):
        self.ql.ensure_state(
            "combat",
            "STATE_1",
            {"ACTION_1": 10.0, "ACTION_2": 35.0, "ACTION_3": 5.0},
        )
        best_act, max_q = self.ql.get_best_action("combat", "STATE_1")
        self.assertEqual(best_act, "ACTION_2")
        self.assertAlmostEqual(max_q, 35.0)

    def test_export_and_load_tables(self):
        self.ql.set_q("roam", "R1", "A1", 10.0)
        self.ql.set_q("farm", "F1", "A2", 20.0)
        self.ql.set_q("combat", "C1", "A3", 30.0)

        dumped = self.ql.export_tables()
        self.assertIn("roam_q", dumped)
        self.assertIn("farm_q", dumped)
        self.assertIn("combat_q", dumped)

        new_ql = QLearningCore()
        new_ql.load_tables(dumped)
        self.assertEqual(new_ql.get_q("roam", "R1", "A1"), 10.0)
        self.assertEqual(new_ql.get_q("farm", "F1", "A2"), 20.0)
        self.assertEqual(new_ql.get_q("combat", "C1", "A3"), 30.0)

    def test_architecture_isolation(self):
        """brain.q_learning must not import hardware, actions, vision, or tactical trees."""
        import brain.q_learning as ql_mod
        forbidden = [
            "subprocess",
            "cv2",
            "ultralytics",
            "control",
            "actions",
            "joystick_controller",
            "claude_macro",
            "realtime_vision",
        ]
        for f in forbidden:
            self.assertFalse(hasattr(ql_mod, f))


if __name__ == "__main__":
    unittest.main()
