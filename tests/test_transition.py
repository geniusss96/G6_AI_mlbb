"""Unit tests for brain/transition.py (Stage 17A: RL Transition Model)."""

import math
import sys
import unittest
from dataclasses import FrozenInstanceError

from brain.transition import Transition


class TestTransitionModel(unittest.TestCase):
    def test_regular_transition_creation(self) -> None:
        """Verifies creating a standard non-terminal transition."""
        t = Transition(
            state_key="SWEET_SPOT_HIGH_COMBO_True",
            action="DIVE_ALL_IN",
            reward=35.0,
            next_state_key="DANGER_CLOSE_MID_COMBO_False",
            done=False,
        )
        self.assertEqual(t.state_key, "SWEET_SPOT_HIGH_COMBO_True")
        self.assertEqual(t.action, "DIVE_ALL_IN")
        self.assertEqual(t.reward, 35.0)
        self.assertEqual(t.next_state_key, "DANGER_CLOSE_MID_COMBO_False")
        self.assertFalse(t.done)

    def test_terminal_transition_with_none_next_state(self) -> None:
        """Verifies terminal transition with next_state_key=None and done=True."""
        t = Transition(
            state_key="CRITICAL_HP",
            action="TACTICAL_RETREAT",
            reward=-40.0,
            next_state_key=None,
            done=True,
        )
        self.assertEqual(t.state_key, "CRITICAL_HP")
        self.assertIsNone(t.next_state_key)
        self.assertTrue(t.done)

    def test_immutability(self) -> None:
        """Transition must be frozen and raise error on attempted modification."""
        t = Transition("S1", "A1", 10.0, "S2", False)
        with self.assertRaises(FrozenInstanceError):
            t.reward = 20.0  # type: ignore

    def test_deterministic_equality_and_hashing(self) -> None:
        """Transitions with identical fields must be equal and hash identically."""
        t1 = Transition("S1", "A1", 15.5, "S2", False)
        t2 = Transition("S1", "A1", 15.5, "S2", False)
        t3 = Transition("S1", "A1", 15.5, "S2", True)

        self.assertEqual(t1, t2)
        self.assertEqual(hash(t1), hash(t2))
        self.assertNotEqual(t1, t3)

    def test_empty_or_whitespace_state_key_rejected(self) -> None:
        """Empty or blank state_key must raise ValueError."""
        with self.assertRaises(ValueError):
            Transition("", "ATTACK", 10.0, "S2", False)
        with self.assertRaises(ValueError):
            Transition("   ", "ATTACK", 10.0, "S2", False)

    def test_empty_or_whitespace_action_rejected(self) -> None:
        """Empty or blank action must raise ValueError."""
        with self.assertRaises(ValueError):
            Transition("S1", "", 10.0, "S2", False)
        with self.assertRaises(ValueError):
            Transition("S1", "  ", 10.0, "S2", False)

    def test_non_finite_reward_rejected(self) -> None:
        """NaN, +inf, -inf rewards must raise ValueError."""
        with self.assertRaises(ValueError):
            Transition("S1", "A1", float("nan"), "S2", False)
        with self.assertRaises(ValueError):
            Transition("S1", "A1", float("inf"), "S2", False)
        with self.assertRaises(ValueError):
            Transition("S1", "A1", float("-inf"), "S2", False)

    def test_type_validation(self) -> None:
        """Invalid types must raise TypeError."""
        # Non-string state_key
        with self.assertRaises(TypeError):
            Transition(123, "A1", 10.0, "S2", False)  # type: ignore

        # Non-string action
        with self.assertRaises(TypeError):
            Transition("S1", None, 10.0, "S2", False)  # type: ignore

        # Non-numeric reward
        with self.assertRaises(TypeError):
            Transition("S1", "A1", "not_a_number", "S2", False)  # type: ignore

        # Non-string, non-None next_state_key
        with self.assertRaises(TypeError):
            Transition("S1", "A1", 10.0, 999, False)  # type: ignore

        # Non-bool done
        with self.assertRaises(TypeError):
            Transition("S1", "A1", 10.0, "S2", 1)  # type: ignore
        with self.assertRaises(TypeError):
            Transition("S1", "A1", 10.0, "S2", "true")  # type: ignore

    def test_architecture_isolation(self) -> None:
        """brain.transition must have zero dependencies on hardware, vision, or Q-core."""
        import brain.transition as trans_mod
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
            self.assertFalse(hasattr(trans_mod, f))


if __name__ == "__main__":
    unittest.main()
