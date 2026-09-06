"""Unit tests for brain/encoder.py (Stage 17B: Tactical State Encoder)."""

import hashlib
import json
import math
import os
import unittest

from brain.encoder import TacticalStateEncoder
from brain.state import TacticalState
from world.models import Vector2


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _make_state(
    timestamp: float = 100.0,
    player_pos: Vector2 = Vector2(772.0, 360.0),
    hp_ratio: float = 1.0,
    is_dead: bool = False,
    s1_ready: bool = True,
    s2_ready: bool = True,
    ult_ready: bool = True,
    can_combo: bool = True,
    enemy_visible: bool = False,
    enemy_count: int = 0,
    enemy_dist: float = None,
    enemy_pos: Vector2 = None,
    enemy_id: int = None,
    minion_count: int = 0,
    minion_dist: float = None,
    minion_pos: Vector2 = None,
    minion_id: int = None,
    turret_near: bool = False,
    near_base: bool = False,
) -> TacticalState:
    return TacticalState(
        timestamp=timestamp,
        player_position=player_pos,
        player_hp_ratio=hp_ratio,
        is_player_dead=is_dead,
        s1_ready=s1_ready,
        s2_ready=s2_ready,
        ultimate_ready=ult_ready,
        can_combo=can_combo,
        enemy_visible=enemy_visible,
        enemy_count=enemy_count,
        nearest_enemy_distance=enemy_dist,
        nearest_enemy_position=enemy_pos,
        minion_count=minion_count,
        nearest_minion_distance=minion_dist,
        nearest_minion_position=minion_pos,
        enemy_near_turret=turret_near,
        is_near_base=near_base,
        nearest_enemy_id=enemy_id,
        nearest_minion_id=minion_id,
    )


class TestTacticalStateEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = TacticalStateEncoder()
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
                "data/q_brain_baseline.json was modified during encoder test!",
            )
        if self.initial_q_brain_hash and os.path.exists(self.q_brain_path):
            self.assertEqual(
                self.initial_q_brain_hash,
                _file_hash(self.q_brain_path),
                "q_brain.json was modified during encoder test!",
            )

    def test_combat_encoding_variations(self) -> None:
        """Verifies combat states across distance, health, combo readiness, and overrides."""
        # 1. Turret danger override
        s_turret = _make_state(enemy_visible=True, enemy_dist=200.0, turret_near=True)
        self.assertEqual(self.encoder.encode_combat(s_turret), "TURRET_DANGER")

        # 2. Critical HP danger override
        s_crit = _make_state(enemy_visible=True, enemy_dist=200.0, hp_ratio=0.25)
        self.assertEqual(self.encoder.encode_combat(s_crit), "CRITICAL_HP_DANGER")

        # 3. DANGER_CLOSE (< 260) with HIGH hp (>= 0.65) and combo=True
        s_danger = _make_state(enemy_visible=True, enemy_dist=200.0, hp_ratio=0.80, can_combo=True)
        self.assertEqual(self.encoder.encode_combat(s_danger), "DANGER_CLOSE_HIGH_COMBO_True")

        # 4. SWEET_SPOT (260..430) with MID hp (< 0.65) and combo=False
        s_sweet = _make_state(enemy_visible=True, enemy_dist=350.0, hp_ratio=0.55, can_combo=False)
        self.assertEqual(self.encoder.encode_combat(s_sweet), "SWEET_SPOT_MID_COMBO_False")

        # 5. CHASE_FAR (> 430) with HIGH hp and combo=False
        s_far = _make_state(enemy_visible=True, enemy_dist=500.0, hp_ratio=0.70, can_combo=False)
        self.assertEqual(self.encoder.encode_combat(s_far), "CHASE_FAR_HIGH_COMBO_False")

    def test_farm_encoding_variations(self) -> None:
        """Verifies farm states across distance buckets, S1 readiness, and health."""
        # 1. CREEP_FAR (> 420), S1 ready, SAFE hp (>= 0.50)
        s_far = _make_state(minion_count=1, minion_dist=500.0, s1_ready=True, hp_ratio=0.80)
        self.assertEqual(self.encoder.encode_farm(s_far), "CREEP_FAR_S1_True_SAFE")

        # 2. CREEP_DANGER_CLOSE (< 260), S1 not ready, SAFE hp
        s_danger = _make_state(minion_count=1, minion_dist=200.0, s1_ready=False, hp_ratio=0.60)
        self.assertEqual(self.encoder.encode_farm(s_danger), "CREEP_DANGER_CLOSE_S1_False_SAFE")

        # 3. CREEP_SWEET_SPOT (260..420), S1 ready, LOW hp (< 0.50)
        s_sweet_low = _make_state(minion_count=1, minion_dist=350.0, s1_ready=True, hp_ratio=0.40)
        self.assertEqual(self.encoder.encode_farm(s_sweet_low), "CREEP_SWEET_SPOT_S1_True_LOW")

    def test_roam_encoding_variations(self) -> None:
        """Verifies roam states across time thresholds."""
        # 1. SCOUT_HOT_ZONE (< 5.0)
        s_hot = _make_state(timestamp=3.0)
        self.assertEqual(self.encoder.encode_roam(s_hot, last_enemy_seen_time=0.0), "SCOUT_HOT_ZONE")

        # 2. SCOUT_MID_SEARCH (5.0 <= t < 15.0)
        s_mid = _make_state(timestamp=10.0)
        self.assertEqual(self.encoder.encode_roam(s_mid, last_enemy_seen_time=0.0), "SCOUT_MID_SEARCH")

        # 3. SCOUT_DEEP_PATROL (>= 15.0)
        s_deep = _make_state(timestamp=20.0)
        self.assertEqual(self.encoder.encode_roam(s_deep, last_enemy_seen_time=0.0), "SCOUT_DEEP_PATROL")

    def test_auto_detection_context(self) -> None:
        """Verifies encode() automatically infers correct context based on state presence."""
        s_combat = _make_state(enemy_visible=True, enemy_dist=300.0, hp_ratio=0.80, can_combo=True)
        self.assertEqual(self.encoder.encode(s_combat), "SWEET_SPOT_HIGH_COMBO_True")

        s_farm = _make_state(minion_count=2, minion_dist=300.0, s1_ready=True, hp_ratio=0.80)
        self.assertEqual(self.encoder.encode(s_farm), "CREEP_SWEET_SPOT_S1_True_SAFE")

        s_roam = _make_state(timestamp=20.0)
        self.assertEqual(self.encoder.encode(s_roam), "SCOUT_DEEP_PATROL")

    def test_deterministic_encoding(self) -> None:
        """Repeated encodings of identical state must yield identical strings."""
        s = _make_state(enemy_visible=True, enemy_dist=350.0, hp_ratio=0.80, can_combo=True)
        key1 = self.encoder.encode(s)
        key2 = self.encoder.encode(s)
        self.assertEqual(key1, key2)

    def test_rejection_of_nan_and_inf(self) -> None:
        """NaN and Inf values must be strictly rejected with ValueError."""
        with self.assertRaises(ValueError):
            self.encoder.encode(_make_state(hp_ratio=float("nan")))

        with self.assertRaises(ValueError):
            self.encoder.encode(_make_state(hp_ratio=float("inf")))

        with self.assertRaises(ValueError):
            self.encoder.encode(_make_state(timestamp=float("nan")))

        with self.assertRaises(ValueError):
            self.encoder.encode(_make_state(enemy_dist=float("nan")))

    def test_q_brain_json_all_keys_reproducible(self) -> None:
        """Verifies that all keys present in existing q_brain.json can be generated by encoder."""
        if not os.path.exists(self.q_brain_path):
            self.skipTest("q_brain.json not present in workspace")

        with open(self.q_brain_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Roam keys
        for key in data.get("roam_q", {}).keys():
            if key == "SCOUT_HOT_ZONE":
                res = self.encoder.encode_roam(_make_state(timestamp=2.0))
            elif key == "SCOUT_MID_SEARCH":
                res = self.encoder.encode_roam(_make_state(timestamp=10.0))
            elif key == "SCOUT_DEEP_PATROL":
                res = self.encoder.encode_roam(_make_state(timestamp=25.0))
            else:
                res = None
            self.assertEqual(res, key)

        # 2. Combat overrides
        self.assertEqual(
            self.encoder.encode_combat(_make_state(turret_near=True)),
            "TURRET_DANGER",
        )

    def test_architecture_isolation(self) -> None:
        """brain.encoder must have zero dependencies on hardware, vision, or Q-core."""
        import brain.encoder as enc_mod
        forbidden = [
            "subprocess",
            "cv2",
            "control",
            "adb",
            "vision",
            "q_learning",
            "rewards",
            "memory",
            "realtime_vision",
        ]
        for f in forbidden:
            self.assertFalse(hasattr(enc_mod, f))


if __name__ == "__main__":
    unittest.main()
