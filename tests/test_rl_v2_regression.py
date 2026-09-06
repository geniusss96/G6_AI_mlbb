"""Regression tests comparing V1 update_q against V2 update_td (Stage 17D)."""

import hashlib
import os
import unittest

from brain.q_learning import QLearningCore
from brain.transition import Transition


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestRLV2Regression(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_path = os.path.join("data", "q_brain_baseline.json")
        self.q_brain_path = "q_brain.json"

        self.initial_baseline_hash = (
            _file_hash(self.baseline_path) if os.path.exists(self.baseline_path) else None
        )
        self.initial_q_brain_hash = (
            _file_hash(self.q_brain_path) if os.path.exists(self.q_brain_path) else None
        )

    def tearDown(self) -> None:
        if self.initial_baseline_hash and os.path.exists(self.baseline_path):
            self.assertEqual(
                self.initial_baseline_hash,
                _file_hash(self.baseline_path),
                "data/q_brain_baseline.json was modified during regression test!",
            )
        if self.initial_q_brain_hash and os.path.exists(self.q_brain_path):
            self.assertEqual(
                self.initial_q_brain_hash,
                _file_hash(self.q_brain_path),
                "q_brain.json was modified during regression test!",
            )

    def test_terminal_transition_equals_v1_update_q(self) -> None:
        """When a transition is terminal (done=True), V2 update_td matches V1 update_q 1:1."""
        ql_v1 = QLearningCore(learning_rate=0.25, discount_factor=0.85)
        ql_v2 = QLearningCore(learning_rate=0.25, discount_factor=0.85)

        old_q = 14.5
        reward = 40.0

        ql_v1.set_q("combat", "COMBAT_STATE", "DIVE_ALL_IN", old_q)
        ql_v2.set_q("combat", "COMBAT_STATE", "DIVE_ALL_IN", old_q)

        # V1 update_q
        v1_result = ql_v1.update_q("combat", "COMBAT_STATE", "DIVE_ALL_IN", reward=reward)

        # V2 update_td with done=True
        t = Transition(
            state_key="COMBAT_STATE",
            action="DIVE_ALL_IN",
            reward=reward,
            next_state_key=None,
            done=True,
        )
        v2_result = ql_v2.update_td(t)

        self.assertEqual(v1_result, v2_result)
        self.assertEqual(
            ql_v1.get_q("combat", "COMBAT_STATE", "DIVE_ALL_IN"),
            ql_v2.get_q("combat", "COMBAT_STATE", "DIVE_ALL_IN"),
        )

    def test_zero_future_value_equals_v1_update_q(self) -> None:
        """When next_state has zero Q values, V2 target equals V1 target (reward)."""
        ql_v1 = QLearningCore(learning_rate=0.20, discount_factor=0.85)
        ql_v2 = QLearningCore(learning_rate=0.20, discount_factor=0.85)

        ql_v1.set_q("farm", "F1", "FARM_APPROACH", 20.0)
        ql_v2.set_q("farm", "F1", "FARM_APPROACH", 20.0)

        # Next state has 0.0 for all actions
        ql_v2.init_state("farm", "F2_EMPTY", {"FARM_APPROACH": 0.0, "FARM_KITE_BACK": 0.0})

        v1_res = ql_v1.update_q("farm", "F1", "FARM_APPROACH", reward=25.0)

        t = Transition("F1", "FARM_APPROACH", 25.0, "F2_EMPTY", False)
        v2_res = ql_v2.update_td(t)

        self.assertEqual(v1_res, v2_res)

    def test_divergence_only_where_future_value_exists(self) -> None:
        """Proves V1 and V2 diverge exclusively when non-zero future Q values exist."""
        ql_v1 = QLearningCore(learning_rate=0.25, discount_factor=0.85)
        ql_v2 = QLearningCore(learning_rate=0.25, discount_factor=0.85)

        ql_v1.set_q("combat", "S_CURR", "SWEET_SPOT_BURST", 10.0)
        ql_v2.set_q("combat", "S_CURR", "SWEET_SPOT_BURST", 10.0)

        # Future state has substantial value
        ql_v2.set_q("combat", "S_NEXT", "DIVE_ALL_IN", 60.0)

        v1_res = ql_v1.update_q("combat", "S_CURR", "SWEET_SPOT_BURST", reward=10.0)
        # In V1: 10.0 + 0.25 * (10.0 - 10.0) = 10.0
        self.assertEqual(v1_res, 10.0)

        t = Transition("S_CURR", "SWEET_SPOT_BURST", 10.0, "S_NEXT", False)
        v2_res = ql_v2.update_td(t)
        # In V2: target = 10.0 + 0.85 * 60.0 = 61.0
        # new_q = 10.0 + 0.25 * (61.0 - 10.0) = 22.75
        self.assertEqual(v2_res, 22.75)
        self.assertNotEqual(v1_res, v2_res)
        self.assertGreater(v2_res, v1_res)


if __name__ == "__main__":
    unittest.main()
