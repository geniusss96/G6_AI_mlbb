"""Unit tests for brain/memory.py (Stage 14D: Brain Memory Extraction)."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from brain.memory import ActionRecord, BrainMemory, Experience


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestBrainMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_path = os.path.join("data", "q_brain_baseline.json")
        self.q_brain_path = "q_brain.json"
        if os.path.exists(self.baseline_path):
            self.initial_baseline_hash = _file_hash(self.baseline_path)
        else:
            self.initial_baseline_hash = None

    def tearDown(self) -> None:
        # Guarantee baseline file was never modified during any test
        if self.initial_baseline_hash is not None and os.path.exists(self.baseline_path):
            current_hash = _file_hash(self.baseline_path)
            self.assertEqual(
                self.initial_baseline_hash,
                current_hash,
                "data/q_brain_baseline.json must remain strictly unmodified!",
            )

    def test_load_existing_format_from_q_brain(self) -> None:
        """Tests reading production q_brain.json schema directly."""
        if not os.path.exists(self.q_brain_path):
            self.skipTest("q_brain.json not present in workspace root")

        memory = BrainMemory(memory_file=self.q_brain_path)
        loaded = memory.load()
        self.assertTrue(loaded)
        self.assertIn("SCOUT_DEEP_PATROL", memory.roam_q_table)
        self.assertIn("DANGER_CLOSE_HIGH_COMBO_True", memory.combat_q_table)
        self.assertGreater(len(memory.best_moments), 0)
        self.assertIsInstance(memory.best_moments[0], Experience)
        self.assertGreater(memory.total_decisions, 0)

    def test_load_baseline_format_without_modifications(self) -> None:
        """Tests reading data/q_brain_baseline.json and ensures it is unchanged."""
        if not os.path.exists(self.baseline_path):
            self.skipTest("data/q_brain_baseline.json not present")

        memory = BrainMemory(memory_file=self.baseline_path)
        loaded = memory.load()
        self.assertTrue(loaded)
        self.assertIn("SCOUT_HOT_ZONE", memory.roam_q_table)
        self.assertIn("CREEP_SWEET_SPOT_S1_True_SAFE", memory.farm_q_table)

        # Confirm baseline hash didn't change
        self.assertEqual(self.initial_baseline_hash, _file_hash(self.baseline_path))

    def test_save_and_roundtrip_in_temp_dir(self) -> None:
        """Tests save, reload, and roundtrip consistency in a safe temp directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = os.path.join(tmp_dir, "test_q_brain.json")
            memory = BrainMemory(memory_file=temp_path)

            memory.roam_q_table = {"SCOUT_HOT_ZONE": {"LANE_ADVANCE": 42.5}}
            memory.farm_q_table = {"CREEP_SAFE": {"FARM_S1_AOE": 20.0}}
            memory.combat_q_table = {"DANGER_CLOSE": {"KITE_AND_POKE": 35.0}}
            memory.total_decisions = 105

            exp = Experience(
                id=1,
                type="BURST_COMBO",
                reward=35.0,
                hp_pct=90,
                actions=("KITE_AND_POKE", "DIVE_ALL_IN"),
                description="Test Combo",
                timestamp="2026-09-06 12:00:00",
            )
            memory.record_moment(exp)

            save_ok = memory.save(timestamp_str="2026-09-06 12:00:00")
            self.assertTrue(save_ok)
            self.assertTrue(os.path.exists(temp_path))

            # Reload into a fresh memory instance
            memory2 = BrainMemory(memory_file=temp_path)
            load_ok = memory2.load()
            self.assertTrue(load_ok)
            self.assertEqual(memory2.roam_q_table, memory.roam_q_table)
            self.assertEqual(memory2.farm_q_table, memory.farm_q_table)
            self.assertEqual(memory2.combat_q_table, memory.combat_q_table)
            self.assertEqual(memory2.total_decisions, 105)
            self.assertEqual(len(memory2.best_moments), 1)
            self.assertEqual(memory2.best_moments[0].actions, ("KITE_AND_POKE", "DIVE_ALL_IN"))
            self.assertEqual(memory2.best_moments[0].reward, 35.0)

    def test_empty_memory_and_missing_file(self) -> None:
        """Tests graceful handling when file is missing or memory is empty."""
        memory = BrainMemory(memory_file="non_existent_file_987654.json")
        self.assertFalse(memory.load())
        self.assertEqual(len(memory.roam_q_table), 0)
        self.assertEqual(len(memory.farm_q_table), 0)
        self.assertEqual(len(memory.combat_q_table), 0)
        self.assertEqual(len(memory.best_moments), 0)
        self.assertEqual(memory.total_decisions, 0)

    def test_reset(self) -> None:
        """Tests reset clearing all state and tables."""
        memory = BrainMemory()
        memory.roam_q_table = {"S": {"A": 1.0}}
        memory.farm_q_table = {"S": {"A": 2.0}}
        memory.combat_q_table = {"S": {"A": 3.0}}
        memory.total_decisions = 50
        memory.log_action("ROAM", "S", "A", 100.0)
        memory.record_moment(
            Experience(1, "KILL", 60.0, 100, ("A",), "desc", "2026-01-01 00:00:00")
        )

        memory.reset()
        self.assertEqual(memory.roam_q_table, {})
        self.assertEqual(memory.farm_q_table, {})
        self.assertEqual(memory.combat_q_table, {})
        self.assertEqual(memory.total_decisions, 0)
        self.assertEqual(len(memory.recent_action_history), 0)
        self.assertEqual(len(memory.best_moments), 0)
        self.assertIsNone(memory.saved_at)

    def test_action_history_sliding_window_limit(self) -> None:
        """Tests that recent_action_history stays capped at 30 items."""
        memory = BrainMemory()
        for i in range(40):
            memory.log_action("COMBAT", f"STATE_{i}", f"ACTION_{i}", timestamp=float(i))

        self.assertEqual(len(memory.recent_action_history), 30)
        self.assertEqual(memory.recent_action_history[0].action, "ACTION_10")
        self.assertEqual(memory.recent_action_history[-1].action, "ACTION_39")

    def test_create_moment_from_history(self) -> None:
        """Tests creating experience moments linked to recent actions without mutating Q-tables."""
        memory = BrainMemory()
        memory.roam_q_table = {"HOT_ZONE": {"LANE_ADVANCE": 10.0}}

        # Log actions at timestamps 100.0, 102.0, 104.0, 106.0
        memory.log_action("COMBAT", "S1", "KITE_AND_POKE", timestamp=100.0)
        memory.log_action("COMBAT", "S2", "SWEET_SPOT_BURST", timestamp=102.0)
        memory.log_action("COMBAT", "S3", "DIVE_ALL_IN", timestamp=104.0)

        # Create moment at time 105.0 (within 8.0s window)
        moment, recent = memory.create_moment_from_history(
            moment_type="HERO_KILL",
            reward=60.0,
            description="Zilong eliminated",
            hp_left_pct=85,
            current_time=105.0,
            timestamp_str="2026-09-06 12:00:00",
        )

        self.assertEqual(moment.id, 1)
        self.assertEqual(moment.type, "HERO_KILL")
        self.assertEqual(moment.reward, 60.0)
        self.assertEqual(moment.hp_pct, 85)
        self.assertEqual(
            moment.actions, ("KITE_AND_POKE", "SWEET_SPOT_BURST", "DIVE_ALL_IN")
        )
        self.assertEqual(len(memory.best_moments), 1)

        # Ensure NO Q-table mutation occurred
        self.assertEqual(memory.roam_q_table["HOT_ZONE"]["LANE_ADVANCE"], 10.0)

    def test_recall_best_tactic(self) -> None:
        """Tests recalling combat tactic from recent best moments."""
        memory = BrainMemory()
        self.assertIsNone(memory.recall_best_tactic(["DIVE_ALL_IN", "KITE_AND_POKE"]))

        moment = Experience(
            id=1,
            type="HERO_KILL",
            reward=60.0,
            hp_pct=90,
            actions=("SCOUT", "DIVE_ALL_IN", "FARM"),
            description="Triumph",
            timestamp="2026-09-06 12:00:00",
        )
        memory.record_moment(moment)

        # DIVE_ALL_IN is in combat actions and should be recalled
        tactic = memory.recall_best_tactic(["DIVE_ALL_IN", "KITE_AND_POKE"])
        self.assertEqual(tactic, "DIVE_ALL_IN")

    def test_deterministic_serialization(self) -> None:
        """Tests that saving twice with identical data yields identical byte strings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "m1.json")
            file2 = os.path.join(tmp_dir, "m2.json")

            m1 = BrainMemory(memory_file=file1)
            m2 = BrainMemory(memory_file=file2)

            tables = {"S": {"A": 1.5, "B": 2.5}}
            m1.roam_q_table = copy.deepcopy(tables)
            m2.roam_q_table = copy.deepcopy(tables)

            fixed_ts = "2026-09-06 12:00:00"
            m1.save(timestamp_str=fixed_ts)
            m2.save(timestamp_str=fixed_ts)

            self.assertEqual(_file_hash(file1), _file_hash(file2))


if __name__ == "__main__":
    import copy
    unittest.main()
