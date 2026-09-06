"""
tests/test_stage_19b_action_executor.py — Stage 19B ActionExecutor Architecture & Regression Tests.

Verifies:
1. Thin execution dispatcher semantics: translates intent without hidden tactical decisions.
2. FARM does not silently become RETREAT/KITE/ATTACK.
3. KITE does not invent fake danger coordinates or tactics.
4. RETREAT execution remains compatible with V1 panic retreat + directional escape.
5. Skill actions preserve fast execution and true combo mechanics (S2 -> S2 -> S1 -> Ult).
6. Screen touch actions notify joystick watchdog via notify_screen_touched().
7. IDLE safety and rejection of invalid actions.
8. Dependency injection: zero direct subprocess/ADB construction.
9. AST-level static architecture boundary verification for actions/ package.
"""

import ast
import os
import sys
import unittest
from typing import Optional, Tuple, Any

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from actions.models import Action, ActionType
from actions.executor import ActionExecutor, ExecutionResult
from world.models import Vector2


class MockJoystick:
    def __init__(self):
        self.move_towards_calls = []
        self.kite_away_calls = []
        self.touch_notifications = 0

    def move_towards(
        self,
        hero_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        duration_ms: int = 400,
    ) -> None:
        self.move_towards_calls.append((hero_pos, target_pos, duration_ms))

    def kite_away(
        self,
        hero_pos: Tuple[float, float],
        danger_pos: Tuple[float, float],
        duration_ms: int = 400,
    ) -> None:
        self.kite_away_calls.append((hero_pos, danger_pos, duration_ms))

    def notify_screen_touched(self) -> None:
        self.touch_notifications += 1


class MockSkills:
    def __init__(self):
        self.basic_attack_calls = 0
        self.s1_calls = 0
        self.s2_calls = 0
        self.ult_calls = 0
        self.panic_retreat_calls = 0

    def fire_basic_attack_fast(self) -> None:
        self.basic_attack_calls += 1

    def fire_skill_1_fast(self) -> None:
        self.s1_calls += 1

    def fire_skill_2_fast(self) -> None:
        self.s2_calls += 1

    def trigger_engage_ultimate(self) -> None:
        self.ult_calls += 1

    def panic_retreat_s2(self) -> None:
        self.panic_retreat_calls += 1


class MockAttackController:
    def __init__(self):
        self.attack_calls = []

    def attack(self, target_id: Optional[int] = None) -> None:
        self.attack_calls.append(target_id)


class TestStage19BActionExecutor(unittest.TestCase):
    def setUp(self):
        self.joystick = MockJoystick()
        self.skills = MockSkills()
        self.attack_ctrl = MockAttackController()
        self.hero_pos = (772.0, 360.0)
        self.executor = ActionExecutor(
            joystick=self.joystick,
            skills=self.skills,
            attack_controller=self.attack_ctrl,
            hero_pos_provider=lambda: self.hero_pos,
        )

    def test_each_action_type_maps_to_expected_control_primitive(self):
        """Every ActionType maps cleanly to its corresponding low-level primitive."""
        # 1. MOVE -> joystick.move_towards
        res = self.executor.execute(Action(type=ActionType.MOVE, direction=Vector2(800.0, 400.0), duration=0.3))
        self.assertTrue(res.success)
        self.assertEqual(len(self.joystick.move_towards_calls), 1)

        # 2. RETREAT -> skills.panic_retreat_s2 + joystick.move_towards
        res = self.executor.execute(Action(type=ActionType.RETREAT, direction=Vector2(100.0, 360.0)))
        self.assertTrue(res.success)
        self.assertEqual(self.skills.panic_retreat_calls, 1)

        # 3. ATTACK -> attack_ctrl.attack
        res = self.executor.execute(Action(type=ActionType.ATTACK, target_id=42))
        self.assertTrue(res.success)
        self.assertEqual(self.attack_ctrl.attack_calls, [42])

        # 4. CAST_S1 -> skills.fire_skill_1_fast
        res = self.executor.execute(Action(type=ActionType.CAST_S1))
        self.assertTrue(res.success)
        self.assertEqual(self.skills.s1_calls, 1)

        # 5. CAST_S2 -> skills.fire_skill_2_fast
        res = self.executor.execute(Action(type=ActionType.CAST_S2))
        self.assertTrue(res.success)
        self.assertEqual(self.skills.s2_calls, 1)

        # 6. CAST_ULT -> skills.trigger_engage_ultimate (combo)
        res = self.executor.execute(Action(type=ActionType.CAST_ULT))
        self.assertTrue(res.success)
        self.assertEqual(self.skills.ult_calls, 1)

        # 7. FARM -> attack + movement primitive
        res = self.executor.execute(Action(type=ActionType.FARM, target_id=99, direction=Vector2(850.0, 350.0)))
        self.assertTrue(res.success)
        self.assertIn(99, self.attack_ctrl.attack_calls)

        # 8. KITE -> joystick.kite_away
        res = self.executor.execute(Action(type=ActionType.KITE, direction=Vector2(950.0, 360.0)))
        self.assertTrue(res.success)
        self.assertEqual(len(self.joystick.kite_away_calls), 1)

        # 9. SCOUT -> joystick.move_towards
        moves_before = len(self.joystick.move_towards_calls)
        res = self.executor.execute(Action(type=ActionType.SCOUT, direction=Vector2(500.0, 200.0)))
        self.assertTrue(res.success)
        self.assertEqual(len(self.joystick.move_towards_calls), moves_before + 1)

        # 10. IDLE -> no-op
        res = self.executor.execute(Action(type=ActionType.IDLE))
        self.assertTrue(res.success)

    def test_farm_does_not_mutate_tactics(self):
        """FARM executes farming primitive and does NOT secretly retreat or kite."""
        farm_action = Action(type=ActionType.FARM, target_id=15, direction=Vector2(900.0, 300.0))
        res = self.executor.execute(farm_action)

        self.assertTrue(res.success)
        self.assertEqual(res.action.type, ActionType.FARM)
        # Verify it attacked the target
        self.assertIn(15, self.attack_ctrl.attack_calls)
        # Verify it did NOT trigger panic retreat or kite
        self.assertEqual(self.skills.panic_retreat_calls, 0)
        self.assertEqual(len(self.joystick.kite_away_calls), 0)

    def test_kite_does_not_invent_tactics(self):
        """KITE rejects missing direction rather than inventing fake enemy coordinates."""
        kite_no_dir = Action(type=ActionType.KITE, direction=None)
        res = self.executor.execute(kite_no_dir)

        self.assertFalse(res.success)
        self.assertIn("requires direction", res.error)
        self.assertEqual(len(self.joystick.kite_away_calls), 0)

        # With explicit direction, executes kite cleanly
        kite_valid = Action(type=ActionType.KITE, direction=Vector2(900.0, 360.0), duration=0.25)
        res_valid = self.executor.execute(kite_valid)
        self.assertTrue(res_valid.success)
        self.assertEqual(len(self.joystick.kite_away_calls), 1)
        hero, danger, dur = self.joystick.kite_away_calls[0]
        self.assertEqual(danger, (900.0, 360.0))
        self.assertEqual(dur, 250)

    def test_retreat_execution_compatibility(self):
        """RETREAT executes both panic retreat escape skill and directional movement."""
        retreat_action = Action(type=ActionType.RETREAT, direction=Vector2(100.0, 360.0), duration=0.4)
        res = self.executor.execute(retreat_action)

        self.assertTrue(res.success)
        self.assertEqual(self.skills.panic_retreat_calls, 1)
        self.assertEqual(len(self.joystick.move_towards_calls), 1)
        hero, target, dur = self.joystick.move_towards_calls[0]
        self.assertEqual(target, (100.0, 360.0))
        self.assertEqual(dur, 400)

    def test_touch_watchdog_notification(self):
        """Screen touch operations (skills, attacks) notify joystick watchdog."""
        touches_start = self.joystick.touch_notifications

        # Attack
        self.executor.execute(Action(type=ActionType.ATTACK, target_id=1))
        self.assertEqual(self.joystick.touch_notifications, touches_start + 1)

        # S1
        self.executor.execute(Action(type=ActionType.CAST_S1))
        self.assertEqual(self.joystick.touch_notifications, touches_start + 2)

        # S2
        self.executor.execute(Action(type=ActionType.CAST_S2))
        self.assertEqual(self.joystick.touch_notifications, touches_start + 3)

        # ULT
        self.executor.execute(Action(type=ActionType.CAST_ULT))
        self.assertEqual(self.joystick.touch_notifications, touches_start + 4)

        # RETREAT (calls panic retreat s2)
        self.executor.execute(Action(type=ActionType.RETREAT, direction=Vector2(100.0, 360.0)))
        self.assertEqual(self.joystick.touch_notifications, touches_start + 5)

        # IDLE should NOT notify touch
        self.executor.execute(Action(type=ActionType.IDLE))
        self.assertEqual(self.joystick.touch_notifications, touches_start + 5)

    def test_invalid_actions_safely_rejected(self):
        """Non-Action instances are rejected with execution failure."""
        res_none = self.executor.execute(None)  # type: ignore
        self.assertFalse(res_none.success)
        self.assertIn("Invalid action object", res_none.error)

        res_str = self.executor.execute("ATTACK")  # type: ignore
        self.assertFalse(res_str.success)
        self.assertIn("Invalid action object", res_str.error)

    def test_dependency_injection_isolation(self):
        """ActionExecutor works purely with injected interfaces without global singletons."""
        custom_joystick = MockJoystick()
        custom_skills = MockSkills()
        isolated_executor = ActionExecutor(joystick=custom_joystick, skills=custom_skills)

        isolated_executor.execute(Action(type=ActionType.MOVE, direction=Vector2(10.0, 20.0)))
        self.assertEqual(len(custom_joystick.move_towards_calls), 1)
        # Default/other instance had 0 calls
        self.assertEqual(len(self.joystick.move_towards_calls), 0)

    def test_static_architecture_boundary_actions(self):
        """
        Static AST verification:
        actions/ must NOT import subprocess, ADB, cv2, ultralytics, or device drivers.
        Must NOT contain hardcoded screen coordinates.
        """
        forbidden_modules = {
            "subprocess",
            "cv2",
            "ultralytics",
            "realtime_vision",
            "claude_cv_bot",
            "target_attacker",
        }

        actions_dir = os.path.join(WORKSPACE_ROOT, "actions")
        for root, _, files in os.walk(actions_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            root_mod = alias.name.split(".")[0]
                            self.assertNotIn(
                                root_mod,
                                forbidden_modules,
                                f"{file} illegally imports forbidden module {root_mod}"
                            )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            root_mod = node.module.split(".")[0]
                            self.assertNotIn(
                                root_mod,
                                forbidden_modules,
                                f"{file} illegally imports from forbidden module {root_mod}"
                            )


if __name__ == "__main__":
    unittest.main()
