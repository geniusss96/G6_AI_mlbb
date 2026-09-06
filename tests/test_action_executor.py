"""
Unit tests for actions.executor (ActionExecutor).
Verifies mapping of abstract Action models to injected control abstractions,
execution results, error handling, absence of subprocess/ADB, and architecture isolation.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from world.models import Vector2
from actions.models import Action, ActionType
from actions.executor import ActionExecutor, ExecutionResult


class FakeJoystick:
    def __init__(self):
        self.move_calls = []
        self.kite_calls = []

    def move_towards(self, hero_pos, target_pos, duration_ms=400):
        self.move_calls.append((hero_pos, target_pos, duration_ms))

    def kite_away(self, hero_pos, danger_pos, duration_ms=400):
        self.kite_calls.append((hero_pos, danger_pos, duration_ms))


class FakeSkills:
    def __init__(self):
        self.basic_attack_calls = 0
        self.s1_calls = 0
        self.s2_calls = 0
        self.ult_calls = 0
        self.panic_retreat_calls = 0

    def fire_basic_attack_fast(self):
        self.basic_attack_calls += 1

    def fire_skill_1_fast(self):
        self.s1_calls += 1

    def fire_skill_2_fast(self):
        self.s2_calls += 1

    def trigger_engage_ultimate(self):
        self.ult_calls += 1

    def panic_retreat_s2(self):
        self.panic_retreat_calls += 1


class FakeAttackController:
    def __init__(self):
        self.targets = []

    def attack(self, target_id=None):
        self.targets.append(target_id)


class TestActionExecutor(unittest.TestCase):
    def setUp(self):
        self.joystick = FakeJoystick()
        self.skills = FakeSkills()
        self.attack_ctrl = FakeAttackController()
        self.executor = ActionExecutor(
            joystick=self.joystick,
            skills=self.skills,
            attack_controller=self.attack_ctrl,
            hero_pos_provider=lambda: (772.0, 360.0),
        )

    def test_move_delegates_to_joystick(self):
        action = Action(type=ActionType.MOVE, direction=Vector2(900.0, 400.0), duration=0.25)
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertTrue(bool(res))
        self.assertEqual(len(self.joystick.move_calls), 1)
        hero, target, dur = self.joystick.move_calls[0]
        self.assertEqual(hero, (772.0, 360.0))
        self.assertEqual(target, (900.0, 400.0))
        self.assertEqual(dur, 250)

    def test_kite_delegates_to_joystick(self):
        action = Action(type=ActionType.KITE, direction=Vector2(1000.0, 350.0))
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(len(self.joystick.kite_calls), 1)
        hero, danger, dur = self.joystick.kite_calls[0]
        self.assertEqual(hero, (772.0, 360.0))
        self.assertEqual(danger, (1000.0, 350.0))

    def test_retreat_delegates_to_skills_and_joystick(self):
        action = Action(type=ActionType.RETREAT, direction=Vector2(200.0, 360.0))
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(self.skills.panic_retreat_calls, 1)
        self.assertEqual(len(self.joystick.move_calls), 1)
        self.assertEqual(self.joystick.move_calls[0][1], (200.0, 360.0))

    def test_attack_delegates_to_attack_controller_with_target_id(self):
        action = Action(type=ActionType.ATTACK, target_id=7)
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(self.attack_ctrl.targets, [7])

    def test_attack_fallback_to_skills_if_no_attack_controller(self):
        executor = ActionExecutor(skills=self.skills)
        action = Action(type=ActionType.ATTACK)
        res = executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(self.skills.basic_attack_calls, 1)

    def test_cast_s1_delegates_to_skills(self):
        action = Action(type=ActionType.CAST_S1)
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(self.skills.s1_calls, 1)

    def test_cast_s2_delegates_to_skills(self):
        action = Action(type=ActionType.CAST_S2)
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(self.skills.s2_calls, 1)

    def test_cast_ult_delegates_to_skills(self):
        action = Action(type=ActionType.CAST_ULT)
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(self.skills.ult_calls, 1)

    def test_farm_delegates_to_attack_and_movement(self):
        action = Action(type=ActionType.FARM, direction=Vector2(850.0, 320.0), target_id=12)
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(self.attack_ctrl.targets, [12])
        self.assertEqual(len(self.joystick.move_calls), 1)

    def test_scout_delegates_to_joystick_move(self):
        action = Action(type=ActionType.SCOUT, direction=Vector2(600.0, 200.0))
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(len(self.joystick.move_calls), 1)

    def test_idle_does_nothing(self):
        action = Action(type=ActionType.IDLE)
        res = self.executor.execute(action)
        self.assertTrue(res.success)
        self.assertEqual(len(self.joystick.move_calls), 0)
        self.assertEqual(self.skills.basic_attack_calls, 0)
        self.assertEqual(self.skills.s1_calls, 0)

    def test_missing_dependency_returns_failure(self):
        executor_bare = ActionExecutor()
        action_move = Action(type=ActionType.MOVE, direction=Vector2(100.0, 100.0))
        res = executor_bare.execute(action_move)
        self.assertFalse(res.success)
        self.assertFalse(bool(res))
        self.assertIsNotNone(res.error)

    def test_control_exception_captured_in_result(self):
        broken_joystick = FakeJoystick()
        broken_joystick.move_towards = MagicMock(side_effect=RuntimeError("Joystick disconnected"))
        executor = ActionExecutor(joystick=broken_joystick)

        action = Action(type=ActionType.MOVE, direction=Vector2(500.0, 500.0))
        res = executor.execute(action)
        self.assertFalse(res.success)
        self.assertIn("Joystick disconnected", res.error)

    def test_can_execute_query(self):
        executor_bare = ActionExecutor()
        self.assertTrue(executor_bare.can_execute(Action(type=ActionType.IDLE)))
        self.assertFalse(executor_bare.can_execute(Action(type=ActionType.MOVE)))
        self.assertFalse(executor_bare.can_execute(Action(type=ActionType.CAST_S1)))

        executor_full = ActionExecutor(joystick=self.joystick, skills=self.skills)
        self.assertTrue(executor_full.can_execute(Action(type=ActionType.MOVE)))
        self.assertTrue(executor_full.can_execute(Action(type=ActionType.CAST_S1)))

    def test_architecture_isolation(self):
        """ActionExecutor must not import subprocess or contain hardcoded button coordinates."""
        import actions.executor as exec_module
        self.assertFalse(hasattr(exec_module, "subprocess"))
        self.assertFalse(hasattr(exec_module, "ADBTransport"))

        # Verify no hardcoded coordinate constants
        for attr in dir(exec_module):
            self.assertNotIn("COORD_", attr)
            self.assertNotIn("BUTTON_", attr)


if __name__ == "__main__":
    unittest.main()
