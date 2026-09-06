"""
Unit tests for brain.v1_adapter (V1BrainAdapter).
Verifies conversion of V2 WorldState into legacy ClaudeRLBrain arguments,
mapping of legacy decision strings into typed Actions, error handling,
target_id/direction preservation, and hardware isolation.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from world.models import Vector2, PlayerState, EnemyState, MinionState, TurretState, WorldState
from vision.hp_detector import HPObservation
from vision.skill_state import SkillState
from actions.models import Action, ActionType
from brain.v1_adapter import V1BrainAdapter


class FakeLegacyBrain:
    def __init__(self):
        self.combat_decision = "KITE_AND_POKE"
        self.farm_decision = "FARM_SWEET_SPOT"
        self.roam_decision = ("LANE_ADVANCE", (950, 340))
        self.last_combat_args = None
        self.last_farm_args = None
        self.last_roam_args = None

    def evaluate_combat_stance(self, dist_enemy, can_combo, is_turret_near, hp_self_ratio=1.0):
        self.last_combat_args = {
            "dist_enemy": dist_enemy,
            "can_combo": can_combo,
            "is_turret_near": is_turret_near,
            "hp_self_ratio": hp_self_ratio,
        }
        return self.combat_decision

    def evaluate_farm_stance(self, dist_farm, s1_ok, hp_self_ratio=1.0):
        self.last_farm_args = {
            "dist_farm": dist_farm,
            "s1_ok": s1_ok,
            "hp_self_ratio": hp_self_ratio,
        }
        return self.farm_decision

    def choose_roam_direction(self, ref_pt, time_since_enemy):
        self.last_roam_args = {
            "ref_pt": ref_pt,
            "time_since_enemy": time_since_enemy,
        }
        return self.roam_decision


class TestV1BrainAdapter(unittest.TestCase):
    def setUp(self):
        self.legacy_brain = FakeLegacyBrain()
        self.adapter = V1BrainAdapter(self.legacy_brain)

    def _make_world_state(
        self,
        enemies=None,
        minions=None,
        turrets=None,
        hp_val=1.0,
        s1=True,
        s2=True,
        ult=True,
        player_visible=True,
    ):
        return WorldState(
            timestamp=100.0,
            player=PlayerState(
                position=Vector2(772.0, 360.0),
                hp=HPObservation(value=hp_val, confidence=1.0, visible=True),
                is_visible=player_visible,
            ),
            enemies=enemies or [],
            minions=minions or [],
            turrets=turrets or [],
            skills=SkillState(s1_ready=s1, s2_ready=s2, ultimate_ready=ult),
            is_near_base=False,
        )

    def test_combat_stance_conversion_arguments(self):
        enemy = EnemyState(position=Vector2(1000.0, 360.0), bbox=(980, 340, 1020, 380), confidence=0.9)
        turret = TurretState(position=Vector2(850.0, 360.0), bbox=(830, 340, 870, 380), confidence=0.9, is_enemy=True)
        ws = self._make_world_state(enemies=[enemy], turrets=[turret], hp_val=0.75, s1=True, s2=False, ult=True)

        action = self.adapter.decide(ws)

        args = self.legacy_brain.last_combat_args
        self.assertIsNotNone(args)
        self.assertAlmostEqual(args["dist_enemy"], 228.0)
        self.assertFalse(args["can_combo"])  # s2 is False -> can_combo False
        self.assertTrue(args["is_turret_near"])  # dist is 78px < 420px
        self.assertAlmostEqual(args["hp_self_ratio"], 0.75)
        self.assertEqual(action.type, ActionType.KITE)

    def test_combat_decisions_to_actions(self):
        enemy = EnemyState(position=Vector2(972.0, 360.0), bbox=(950, 340, 990, 380), confidence=0.9)
        ws = self._make_world_state(enemies=[enemy])

        # KITE_AND_POKE -> KITE
        self.legacy_brain.combat_decision = "KITE_AND_POKE"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.KITE)
        self.assertEqual(res.direction, enemy.position)

        # TURRET_RETREAT -> RETREAT
        self.legacy_brain.combat_decision = "TURRET_RETREAT"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.RETREAT)
        self.assertIsNotNone(res.direction)

        # TACTICAL_RETREAT -> RETREAT
        self.legacy_brain.combat_decision = "TACTICAL_RETREAT"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.RETREAT)

        # SWEET_SPOT_BURST -> ATTACK
        self.legacy_brain.combat_decision = "SWEET_SPOT_BURST"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.ATTACK)
        self.assertIsNotNone(res.direction)

        # DIVE_ALL_IN -> CAST_ULT
        self.legacy_brain.combat_decision = "DIVE_ALL_IN"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.CAST_ULT)
        self.assertEqual(res.direction, enemy.position)

    def test_farm_decisions_to_actions(self):
        minion = MinionState(position=Vector2(900.0, 360.0), bbox=(880, 340, 920, 380), confidence=0.8, is_enemy=True)
        ws = self._make_world_state(minions=[minion])

        # FARM_APPROACH -> MOVE
        self.legacy_brain.farm_decision = "FARM_APPROACH"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.MOVE)
        self.assertEqual(res.direction, minion.position)

        # FARM_KITE_BACK -> KITE
        self.legacy_brain.farm_decision = "FARM_KITE_BACK"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.KITE)
        self.assertEqual(res.direction, minion.position)

        # FARM_S1_AOE -> CAST_S1
        self.legacy_brain.farm_decision = "FARM_S1_AOE"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.CAST_S1)

        # FARM_SWEET_SPOT -> FARM
        self.legacy_brain.farm_decision = "FARM_SWEET_SPOT"
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.FARM)
        self.assertIsNotNone(res.direction)

    def test_roam_decision_to_scout_action(self):
        ws = self._make_world_state()  # No enemies, no minions
        self.legacy_brain.roam_decision = ("RIVER_SCOUT", (850, 420))
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.SCOUT)
        self.assertEqual(res.direction, Vector2(850.0, 420.0))

    def test_player_dead_returns_idle(self):
        ws = self._make_world_state(player_visible=False)
        res = self.adapter.decide(ws)
        self.assertEqual(res.type, ActionType.IDLE)

    def test_invalid_input_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.adapter.decide(None)  # type: ignore

        with self.assertRaises(TypeError):
            self.adapter.decide({"some": "dict"})  # type: ignore

    def test_architecture_isolation(self):
        """Adapter must not import hardware or optical processing packages."""
        import brain.v1_adapter as adapter_module
        forbidden = ["subprocess", "cv2", "ultralytics", "control", "claude_macro", "joystick_controller"]
        for mod in forbidden:
            self.assertFalse(hasattr(adapter_module, mod))


if __name__ == "__main__":
    unittest.main()
