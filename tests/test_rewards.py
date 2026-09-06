"""
Unit tests for brain.rewards (Reward, RewardCalculator).
Verifies exact V1 numerical reward values, event mapping, determinism,
absence of Q-table mutation, and architecture isolation.
"""

import os
import sys
import unittest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from world.events import EventType, GameEvent
from brain.rewards import Reward, RewardCalculator


class TestRewardCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = RewardCalculator()

    def test_hero_kill_event_rewards(self):
        ev = GameEvent(
            type=EventType.HERO_KILL,
            timestamp=10.0,
            entity_id=1,
            confidence=0.95,
            evidence=("optical_death",),
        )
        rewards = self.calc.for_event(ev)
        self.assertEqual(len(rewards), 2)

        combat_r = next(r for r in rewards if r.domain == "combat")
        roam_r = next(r for r in rewards if r.domain == "roam")

        self.assertEqual(combat_r.value, 60.0)
        self.assertEqual(combat_r.reason, "KILLED ENEMY HERO")
        self.assertEqual(combat_r.source, "hero_kill")

        self.assertEqual(roam_r.value, 40.0)
        self.assertEqual(roam_r.reason, "ROUTE LED TO KILL")
        self.assertEqual(roam_r.source, "hero_kill")

    def test_player_death_event_rewards(self):
        ev = GameEvent(
            type=EventType.PLAYER_DEATH,
            timestamp=20.0,
            entity_id=None,
            confidence=1.0,
            evidence=("hp_zero",),
        )
        rewards = self.calc.for_event(ev)
        self.assertEqual(len(rewards), 2)

        combat_r = next(r for r in rewards if r.domain == "combat")
        roam_r = next(r for r in rewards if r.domain == "roam")

        self.assertEqual(combat_r.value, -40.0)
        self.assertEqual(combat_r.reason, "DIED IN COMBAT")

        self.assertEqual(roam_r.value, -30.0)
        self.assertEqual(roam_r.reason, "DIED ON ROUTE")

    def test_creep_kill_event_rewards(self):
        ev = GameEvent(
            type=EventType.CREEP_KILL,
            timestamp=30.0,
            entity_id=5,
            confidence=0.90,
            evidence=("hp_zero",),
        )
        rewards = self.calc.for_event(ev)
        self.assertEqual(len(rewards), 2)

        farm_r = next(r for r in rewards if r.domain == "farm")
        roam_r = next(r for r in rewards if r.domain == "roam")

        self.assertEqual(farm_r.value, 25.0)
        self.assertEqual(farm_r.reason, "CREEP SLAIN (LAST HIT)")

        self.assertEqual(roam_r.value, 15.0)
        self.assertEqual(roam_r.reason, "FOUND JUNGLE/LANE FARM")

    def test_target_entered_event_reward(self):
        ev = GameEvent(
            type=EventType.TARGET_ENTERED,
            timestamp=40.0,
            entity_id=3,
            confidence=0.85,
            evidence=("new_track",),
        )
        rewards = self.calc.for_event(ev)
        self.assertEqual(len(rewards), 1)
        r = rewards[0]
        self.assertEqual(r.value, 30.0)
        self.assertEqual(r.reason, "FOUND ENEMY HERO")
        self.assertEqual(r.domain, "roam")

    def test_target_lost_event_yields_no_rewards(self):
        ev = GameEvent(
            type=EventType.TARGET_LOST,
            timestamp=50.0,
            entity_id=3,
            confidence=0.85,
            evidence=("lost",),
        )
        rewards = self.calc.for_event(ev)
        self.assertEqual(rewards, [])

    def test_contextual_rewards(self):
        turret_pen = self.calc.for_turret_danger()
        self.assertEqual(turret_pen.value, -40.0)
        self.assertEqual(turret_pen.domain, "combat")
        self.assertEqual(turret_pen.reason, "WALKED INTO TURRET")

        burst_rew = self.calc.for_burst_combo()
        self.assertEqual(burst_rew.value, 35.0)
        self.assertEqual(burst_rew.domain, "combat")

        s1_farm = self.calc.for_s1_aoe_wave()
        self.assertEqual(s1_farm.value, 20.0)
        self.assertEqual(s1_farm.domain, "farm")

        wall_pen = self.calc.for_wall_collision()
        self.assertEqual(wall_pen.value, -25.0)
        self.assertEqual(wall_pen.domain, "roam")

        idle_pen = self.calc.for_idle_penalty()
        self.assertEqual(idle_pen.value, -3.0)
        self.assertEqual(idle_pen.domain, "roam")

    def test_custom_reward(self):
        custom = self.calc.for_custom(12.5, reason="CUSTOM TEST", domain="farm")
        self.assertEqual(custom.value, 12.5)
        self.assertEqual(custom.reason, "CUSTOM TEST")
        self.assertEqual(custom.domain, "farm")

    def test_deterministic_calculation(self):
        ev = GameEvent(
            type=EventType.HERO_KILL,
            timestamp=100.0,
            entity_id=7,
            confidence=0.99,
            evidence=("test",),
        )
        r1 = self.calc.for_event(ev)
        r2 = self.calc.for_event(ev)
        self.assertEqual(r1, r2)

    def test_immutability(self):
        r = Reward(value=10.0, reason="test", source="test")
        with self.assertRaises(Exception):
            r.value = 20.0  # type: ignore

    def test_architecture_isolation(self):
        """brain.rewards must not import hardware, vision, or QLearningCore."""
        import brain.rewards as rew_mod
        forbidden = [
            "subprocess",
            "cv2",
            "ultralytics",
            "control",
            "claude_macro",
            "joystick_controller",
            "realtime_vision",
            "QLearningCore",
        ]
        for f in forbidden:
            self.assertFalse(hasattr(rew_mod, f))


if __name__ == "__main__":
    unittest.main()
