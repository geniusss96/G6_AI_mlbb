"""Unit and regression tests for brain/replay.py (Stage 16C: Experience Replay Separation)."""

import hashlib
import os
import unittest
from unittest.mock import MagicMock

from brain.memory import ActionRecord, BrainMemory, Experience
from brain.q_learning import QLearningCore
from brain.replay import ExperienceReplay
from claude_brain import ClaudeRLBrain


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestExperienceReplay(unittest.TestCase):
    def setUp(self) -> None:
        self.replay = ExperienceReplay()
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
                "data/q_brain_baseline.json was modified during replay test!",
            )
        if self.initial_q_brain_hash and os.path.exists(self.q_brain_path):
            self.assertEqual(
                self.initial_q_brain_hash,
                _file_hash(self.q_brain_path),
                "q_brain.json was modified during replay test!",
            )

    def test_empty_experiences_returns_zero(self) -> None:
        """Empty input list produces zero updates and does not mutate table."""
        ql = QLearningCore()
        count = self.replay.replay_moment([], reward=60.0, q_learning=ql)
        self.assertEqual(count, 0)

        count_batch = self.replay.replay([], q_learning=ql)
        self.assertEqual(count_batch, 0)

    def test_single_action_replay(self) -> None:
        """Single action record receives expected V1 replay update: Q + 0.35*(R - Q)."""
        ql = QLearningCore()
        ql.set_q("combat", "HOT_COMBAT", "KITE_AND_POKE", 10.0)

        record = ActionRecord("COMBAT", "HOT_COMBAT", "KITE_AND_POKE", 100.0)
        # old_q = 10.0, reward = 60.0, alpha = 0.35 -> new_q = 10.0 + 0.35 * (60.0 - 10.0) = 27.5
        count = self.replay.replay_moment([record], reward=60.0, q_learning=ql)

        self.assertEqual(count, 1)
        self.assertEqual(ql.get_q("combat", "HOT_COMBAT", "KITE_AND_POKE"), 27.5)

    def test_multiple_actions_deterministic_order(self) -> None:
        """Sequence of multiple actions are replayed in deterministic order."""
        ql = QLearningCore()
        ql.set_q("roam", "ROAM_STATE", "LANE_ADVANCE", 20.0)
        ql.set_q("combat", "COMBAT_STATE", "SWEET_SPOT_BURST", 30.0)
        ql.set_q("combat", "COMBAT_STATE", "DIVE_ALL_IN", 40.0)

        chain = [
            ActionRecord("ROAM", "ROAM_STATE", "LANE_ADVANCE", 10.0),
            ActionRecord("COMBAT", "COMBAT_STATE", "SWEET_SPOT_BURST", 12.0),
            ActionRecord("COMBAT", "COMBAT_STATE", "DIVE_ALL_IN", 14.0),
        ]

        count = self.replay.replay_moment(chain, reward=80.0, q_learning=ql)
        self.assertEqual(count, 3)

        # roam: 20 + 0.35*(80-20) = 41.0
        self.assertEqual(ql.get_q("roam", "ROAM_STATE", "LANE_ADVANCE"), 41.0)
        # combat s1: 30 + 0.35*(80-30) = 47.5
        self.assertEqual(ql.get_q("combat", "COMBAT_STATE", "SWEET_SPOT_BURST"), 47.5)
        # combat s2: 40 + 0.35*(80-40) = 54.0
        self.assertEqual(ql.get_q("combat", "COMBAT_STATE", "DIVE_ALL_IN"), 54.0)

    def test_update_q_called_delegation_no_duplicate_math(self) -> None:
        """Verifies ExperienceReplay strictly delegates Q mutations to QLearningCore.update_q."""
        mock_ql = MagicMock(spec=QLearningCore)
        mock_ql.get_table.return_value = {"S": {"A": 5.0}}

        record = ActionRecord("COMBAT", "S", "A", 1.0)
        self.replay.replay_moment([record], reward=50.0, q_learning=mock_ql)

        mock_ql.update_q.assert_called_once_with(
            table_name="combat",
            state_key="S",
            action="A",
            reward=50.0,
            alpha=0.35,
        )

    def test_only_existing_guard_matches_v1(self) -> None:
        """In V1, only existing states/actions in the Q-table receive replay updates."""
        ql = QLearningCore()
        # "MISSING_ACTION" is not in table
        ql.set_q("combat", "STATE_X", "KNOWN_ACTION", 10.0)

        record1 = ActionRecord("COMBAT", "STATE_X", "KNOWN_ACTION", 1.0)
        record2 = ActionRecord("COMBAT", "STATE_X", "MISSING_ACTION", 2.0)
        record3 = ActionRecord("COMBAT", "MISSING_STATE", "ANY_ACTION", 3.0)

        count = self.replay.replay_moment([record1, record2, record3], reward=50.0, q_learning=ql, only_existing=True)
        self.assertEqual(count, 1)
        self.assertFalse(ql.has_state("combat", "MISSING_STATE"))
        self.assertEqual(ql.get_q("combat", "STATE_X", "MISSING_ACTION"), 0.0)

    def test_v1_regression_behavior_parity(self) -> None:
        """Proves exact 1:1 mathematical equivalence with V1 record_best_moment replay."""
        # Initialize isolated V1 ClaudeRLBrain
        v1 = ClaudeRLBrain(memory_file="__fake_replay_reg__.json")
        v1.save_memory = lambda: None  # No disk I/O

        # Initialize QLearningCore with identical tables
        ql = QLearningCore()

        initial_tables = {
            "S_ROAM": {"LANE_ADVANCE": 15.0},
            "S_COMBAT": {"KITE_AND_POKE": 25.0, "DIVE_ALL_IN": 35.0},
        }
        v1.roam_q_table = {"S_ROAM": {"LANE_ADVANCE": 15.0}}
        v1.combat_q_table = {"S_COMBAT": {"KITE_AND_POKE": 25.0, "DIVE_ALL_IN": 35.0}}

        ql.roam_q_table = {"S_ROAM": {"LANE_ADVANCE": 15.0}}
        ql.combat_q_table = {"S_COMBAT": {"KITE_AND_POKE": 25.0, "DIVE_ALL_IN": 35.0}}

        # Setup 3 recent actions in V1
        action_history = [
            {"category": "ROAM", "state": "S_ROAM", "action": "LANE_ADVANCE", "time": 100.0},
            {"category": "COMBAT", "state": "S_COMBAT", "action": "KITE_AND_POKE", "time": 102.0},
            {"category": "COMBAT", "state": "S_COMBAT", "action": "DIVE_ALL_IN", "time": 104.0},
        ]
        v1.recent_action_history = list(action_history)

        # Run V1 replay via record_best_moment
        # In V1, now = time.time(), so we patch time.time to 105.0 (< 8.0s delta)
        import time
        from unittest.mock import patch

        with patch("time.time", return_value=105.0):
            v1.record_best_moment(
                moment_type="BURST_COMBO",
                reward=70.0,
                description="Test Triumph",
                hp_left_pct=90,
            )

        # Run V2 ExperienceReplay on the same action history
        records = [
            ActionRecord(a["category"], a["state"], a["action"], a["time"])
            for a in action_history
        ]
        self.replay.replay_moment(records, reward=70.0, q_learning=ql)

        # Assert exact 1:1 match across all updated Q-values
        self.assertEqual(
            ql.get_q("roam", "S_ROAM", "LANE_ADVANCE"),
            v1.roam_q_table["S_ROAM"]["LANE_ADVANCE"],
        )
        self.assertEqual(
            ql.get_q("combat", "S_COMBAT", "KITE_AND_POKE"),
            v1.combat_q_table["S_COMBAT"]["KITE_AND_POKE"],
        )
        self.assertEqual(
            ql.get_q("combat", "S_COMBAT", "DIVE_ALL_IN"),
            v1.combat_q_table["S_COMBAT"]["DIVE_ALL_IN"],
        )

    def test_batch_replay_with_tuples(self) -> None:
        """Verifies replay() with batch tuples (record, individual_reward)."""
        ql = QLearningCore()
        ql.set_q("farm", "F_STATE", "FARM_APPROACH", 10.0)
        ql.set_q("farm", "F_STATE", "FARM_S1_AOE", 20.0)

        batch = [
            (ActionRecord("FARM", "F_STATE", "FARM_APPROACH", 1.0), 30.0),
            (ActionRecord("FARM", "F_STATE", "FARM_S1_AOE", 2.0), 50.0),
        ]

        count = self.replay.replay(batch, q_learning=ql)
        self.assertEqual(count, 2)
        # 10 + 0.35*(30 - 10) = 17.0
        self.assertEqual(ql.get_q("farm", "F_STATE", "FARM_APPROACH"), 17.0)
        # 20 + 0.35*(50 - 20) = 30.5
        self.assertEqual(ql.get_q("farm", "F_STATE", "FARM_S1_AOE"), 30.5)


if __name__ == "__main__":
    unittest.main()
