"""
Unit tests for brain.state (TacticalState, TacticalStateBuilder).

Verifies extraction of distilled numerical features from WorldState,
absence of tactical decisions, determinism, architecture isolation,
and correct separation of:
    LOST != DEAD
    unavailable HP != critical HP
"""

import os
import sys
import unittest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from world.models import (
    Vector2,
    PlayerState,
    EnemyState,
    MinionState,
    TurretState,
    WorldState,
)
from vision.hp_detector import HPObservation
from vision.skill_state import SkillState
from brain.state import TacticalState, TacticalStateBuilder


class TestTacticalStateBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = TacticalStateBuilder()

    def test_player_hp_and_position_extraction(self):
        ws = WorldState(
            timestamp=12.5,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(
                    value=0.45,
                    confidence=0.9,
                    visible=True,
                ),
                is_visible=True,
            ),
        )

        ts = self.builder.build(ws)

        self.assertEqual(ts.timestamp, 12.5)
        self.assertEqual(
            ts.player_position,
            Vector2(500.0, 400.0),
        )
        self.assertAlmostEqual(
            ts.player_hp_ratio,
            0.45,
        )
        self.assertTrue(ts.is_player_visible)
        self.assertTrue(ts.is_player_hp_visible)
        self.assertFalse(ts.is_player_dead)

    def test_dead_player_flag(self):
        """
        Explicit death flag means the player is dead even when the HP bar
        is unavailable.

        This deliberately separates confirmed death from temporary loss
        of visibility.
        """
        ws = WorldState(
            timestamp=15.0,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(
                    value=1.0,
                    confidence=0.0,
                    visible=False,
                ),
                is_visible=False,
                is_dead=True,
            ),
        )

        ts = self.builder.build(ws)

        self.assertTrue(ts.is_player_dead)
        self.assertFalse(ts.is_player_visible)
        self.assertFalse(ts.is_player_hp_visible)

    def test_unavailable_hp_is_not_critical(self):
        """
        Missing/unavailable HP must not be interpreted as real critical HP.
        """
        ws = WorldState(
            timestamp=15.5,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(
                    value=1.0,
                    confidence=0.0,
                    visible=False,
                ),
                is_visible=True,
                is_dead=False,
            ),
        )

        ts = self.builder.build(ws)

        self.assertFalse(ts.is_player_hp_visible)
        self.assertFalse(ts.is_player_dead)
        self.assertEqual(ts.player_hp_ratio, 1.0)

    def test_real_low_hp_is_preserved(self):
        """
        When HP is genuinely visible, a real low HP value must remain intact.
        """
        ws = WorldState(
            timestamp=15.75,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(
                    value=0.10,
                    confidence=1.0,
                    visible=True,
                ),
                is_visible=True,
                is_dead=False,
            ),
        )

        ts = self.builder.build(ws)

        self.assertTrue(ts.is_player_hp_visible)
        self.assertAlmostEqual(ts.player_hp_ratio, 0.10)
        self.assertFalse(ts.is_player_dead)

    def test_lost_player_is_not_dead(self):
        """
        LOST != DEAD:
        Missing vision / temporary loss must never mean player death.
        """
        ws = WorldState(
            timestamp=16.0,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(
                    value=0.85,
                    confidence=0.7,
                    visible=True,
                ),
                is_visible=False,
                is_dead=False,
            ),
        )

        ts = self.builder.build(ws)

        self.assertFalse(
            ts.is_player_dead,
            "LOST != DEAD: temporary loss must not mean death",
        )
        self.assertFalse(ts.is_player_visible)
        self.assertTrue(ts.is_player_hp_visible)
        self.assertAlmostEqual(
            ts.player_hp_ratio,
            0.85,
        )

    def test_optional_player_fallback(self):
        ws = WorldState(
            timestamp=20.0,
            player=None,
        )

        ts = self.builder.build(ws)

        self.assertEqual(
            ts.player_position,
            Vector2(772.0, 360.0),
        )
        self.assertEqual(
            ts.player_hp_ratio,
            1.0,
        )
        self.assertFalse(ts.is_player_dead)
        self.assertFalse(ts.is_player_visible)
        self.assertFalse(ts.is_player_hp_visible)

    def test_nearest_enemy_and_enemy_count(self):
        p = Vector2(700.0, 300.0)

        e1 = EnemyState(
            position=Vector2(900.0, 300.0),
            bbox=(880, 280, 920, 320),
            confidence=0.9,
        )

        e2 = EnemyState(
            position=Vector2(1100.0, 300.0),
            bbox=(1080, 280, 1120, 320),
            confidence=0.8,
        )

        ws = WorldState(
            timestamp=25.0,
            player=PlayerState(
                position=p,
                hp=HPObservation(
                    value=1.0,
                    confidence=1.0,
                    visible=True,
                ),
            ),
            enemies=[e1, e2],
        )

        ts = self.builder.build(ws)

        self.assertTrue(ts.enemy_visible)
        self.assertEqual(ts.enemy_count, 2)
        self.assertAlmostEqual(
            ts.nearest_enemy_distance,
            200.0,
        )
        self.assertEqual(
            ts.nearest_enemy_position,
            Vector2(900.0, 300.0),
        )

    def test_empty_enemy_list(self):
        ws = WorldState(
            timestamp=30.0,
            enemies=[],
        )

        ts = self.builder.build(ws)

        self.assertFalse(ts.enemy_visible)
        self.assertEqual(ts.enemy_count, 0)
        self.assertIsNone(ts.nearest_enemy_distance)
        self.assertIsNone(ts.nearest_enemy_position)

    def test_nearest_minion_and_minion_count(self):
        p = Vector2(700.0, 300.0)

        m_enemy = MinionState(
            position=Vector2(850.0, 300.0),
            bbox=(830, 280, 870, 320),
            confidence=0.8,
            is_enemy=True,
        )

        m_ally = MinionState(
            position=Vector2(600.0, 300.0),
            bbox=(580, 280, 620, 320),
            confidence=0.8,
            is_enemy=False,
        )

        ws = WorldState(
            timestamp=35.0,
            player=PlayerState(
                position=p,
                hp=HPObservation(
                    value=1.0,
                    confidence=1.0,
                    visible=True,
                ),
            ),
            minions=[m_enemy, m_ally],
        )

        ts = self.builder.build(ws)

        self.assertEqual(ts.minion_count, 1)
        self.assertAlmostEqual(
            ts.nearest_minion_distance,
            150.0,
        )
        self.assertEqual(
            ts.nearest_minion_position,
            Vector2(850.0, 300.0),
        )

    def test_empty_minion_list(self):
        ws = WorldState(
            timestamp=40.0,
            minions=[],
        )

        ts = self.builder.build(ws)

        self.assertEqual(ts.minion_count, 0)
        self.assertIsNone(ts.nearest_minion_distance)
        self.assertIsNone(ts.nearest_minion_position)

    def test_skill_readiness_and_combo_flag(self):
        ws = WorldState(
            timestamp=45.0,
            skills=SkillState(
                s1_ready=True,
                s2_ready=False,
                ultimate_ready=True,
            ),
        )

        ts = self.builder.build(ws)

        self.assertTrue(ts.s1_ready)
        self.assertFalse(ts.s2_ready)
        self.assertTrue(ts.ultimate_ready)
        self.assertFalse(ts.can_combo)

        ws2 = WorldState(
            timestamp=46.0,
            skills=SkillState(
                s1_ready=False,
                s2_ready=True,
                ultimate_ready=True,
            ),
        )

        ts2 = self.builder.build(ws2)

        self.assertTrue(ts2.can_combo)

    def test_turret_proximity(self):
        p = Vector2(700.0, 300.0)

        t_danger = TurretState(
            position=Vector2(1000.0, 300.0),
            bbox=(980, 280, 1020, 320),
            confidence=0.9,
            is_enemy=True,
        )

        ws_danger = WorldState(
            timestamp=50.0,
            player=PlayerState(
                position=p,
                hp=HPObservation(
                    value=1.0,
                    confidence=1.0,
                    visible=True,
                ),
            ),
            turrets=[t_danger],
        )

        ts_danger = self.builder.build(ws_danger)

        self.assertTrue(ts_danger.enemy_near_turret)

        t_safe = TurretState(
            position=Vector2(1200.0, 300.0),
            bbox=(1180, 280, 1220, 320),
            confidence=0.9,
            is_enemy=True,
        )

        ws_safe = WorldState(
            timestamp=51.0,
            player=PlayerState(
                position=p,
                hp=HPObservation(
                    value=1.0,
                    confidence=1.0,
                    visible=True,
                ),
            ),
            turrets=[t_safe],
        )

        ts_safe = self.builder.build(ws_safe)

        self.assertFalse(ts_safe.enemy_near_turret)

    def test_immutability(self):
        ws = WorldState(timestamp=60.0)
        ts = self.builder.build(ws)

        with self.assertRaises(Exception):
            ts.timestamp = 999.0  # type: ignore

    def test_deterministic_conversion(self):
        ws = WorldState(
            timestamp=70.0,
            enemies=[
                EnemyState(
                    position=Vector2(900.0, 300.0),
                    bbox=(880, 280, 920, 320),
                    confidence=0.9,
                )
            ],
        )

        ts1 = self.builder.build(ws)
        ts2 = self.builder.build(ws)

        self.assertEqual(ts1, ts2)

    def test_architecture_isolation(self):
        """
        brain.state must not import hardware, actions.executor,
        OpenCV, or YOLO.
        """
        import brain.state as state_module

        forbidden = [
            "subprocess",
            "cv2",
            "ultralytics",
            "control",
            "actions.executor",
            "claude_macro",
            "joystick_controller",
            "realtime_vision",
        ]

        for f in forbidden:
            self.assertFalse(
                hasattr(state_module, f),
                f"Forbidden dependency detected: {f}",
            )


if __name__ == "__main__":
    unittest.main()
