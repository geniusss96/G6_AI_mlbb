"""
Unit tests for brain.tactical (TacticalBrain).
Verifies the contract WorldState -> TacticalBrain.decide() -> Action,
return types, input validation, determinism, and architectural isolation.
"""

import os
import sys
import unittest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from world.models import Vector2, PlayerState, WorldState
from vision.hp_detector import HPObservation
from vision.skill_state import SkillState
from actions.models import Action, ActionType
from brain.tactical import TacticalBrain


class TestTacticalBrainContract(unittest.TestCase):
    def setUp(self):
        self.brain = TacticalBrain()
        self.sample_world_state = WorldState(
            timestamp=100.0,
            player=PlayerState(
                position=Vector2(772.0, 360.0),
                hp=HPObservation(value=1.0, confidence=1.0, visible=True),
                is_visible=True,
            ),
            enemies=[],
            minions=[],
            turrets=[],
            skills=SkillState(s1_ready=True, s2_ready=True, ultimate_ready=True),
            is_near_base=False,
        )

    def test_decide_returns_action_instance(self):
        """decide() must return an Action object."""
        action = self.brain.decide(self.sample_world_state)
        self.assertIsInstance(action, Action)
        self.assertIsInstance(action.type, ActionType)
        self.assertEqual(action.type, ActionType.IDLE)

    def test_decide_input_validation(self):
        """decide() must reject inputs that are not WorldState instances."""
        with self.assertRaises(TypeError):
            self.brain.decide(None)  # type: ignore

        with self.assertRaises(TypeError):
            self.brain.decide({"timestamp": 100.0})  # type: ignore

        with self.assertRaises(TypeError):
            self.brain.decide("raw_frame_data")  # type: ignore

    def test_deterministic_behavior(self):
        """decide() must be completely deterministic for identical inputs."""
        action1 = self.brain.decide(self.sample_world_state)
        action2 = self.brain.decide(self.sample_world_state)
        self.assertEqual(action1, action2)

    def test_architecture_isolation(self):
        """
        TacticalBrain must NOT import control, adb, joystick, macros,
        vision capture/detectors, OpenCV, or subprocess.
        """
        import brain.tactical as tactical_module

        # Module attributes / globals
        forbidden_modules = [
            "subprocess",
            "cv2",
            "ultralytics",
            "claude_macro",
            "joystick_controller",
            "realtime_vision",
            "control",
        ]

        for forbidden in forbidden_modules:
            self.assertFalse(
                hasattr(tactical_module, forbidden),
                f"brain.tactical must not expose or import '{forbidden}'"
            )

        # Inspect imported modules in sys.modules loaded via tactical_module
        with open(tactical_module.__file__, "r", encoding="utf-8") as f:
            source = f.read()

        for term in ["subprocess", "cv2", "ultralytics", "claude_macro", "joystick_controller", "realtime_vision"]:
            self.assertNotIn(f"import {term}", source)
            self.assertNotIn(f"from {term}", source)


if __name__ == "__main__":
    unittest.main()
