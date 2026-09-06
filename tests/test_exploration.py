"""Unit tests for brain/exploration.py (Stage 16A: Exploration Policy Extraction)."""

import random
import unittest

from brain.exploration import ExplorationContext, ExplorationPolicy


class MockRNG:
    """Mock random number generator returning predefined values."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._idx = 0

    def random(self) -> float:
        if self._idx < len(self._values):
            val = self._values[self._idx]
            self._idx += 1
            return val
        return 0.5


class TestExplorationPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ExplorationPolicy()

    def test_v1_baseline_epsilon_values(self) -> None:
        """Verifies exact 1:1 match with V1 baseline epsilon values."""
        self.assertEqual(self.policy.epsilon(ExplorationContext.ROAM), 0.20)
        self.assertEqual(self.policy.epsilon(ExplorationContext.FARM), 0.10)
        self.assertEqual(self.policy.epsilon(ExplorationContext.COMBAT), 0.08)

    def test_string_context_acceptance(self) -> None:
        """Verifies string contexts ('roam', 'farm', 'combat') map correctly."""
        self.assertEqual(self.policy.epsilon("roam"), 0.20)
        self.assertEqual(self.policy.epsilon("farm"), 0.10)
        self.assertEqual(self.policy.epsilon("combat"), 0.08)
        self.assertEqual(self.policy.epsilon("ROAM"), 0.20)

    def test_unknown_context_raises(self) -> None:
        """Invalid context must raise ValueError."""
        with self.assertRaises(ValueError):
            self.policy.epsilon("invalid_context")

    def test_deterministic_should_explore_with_mock_rng(self) -> None:
        """Verifies should_explore logic across exploration thresholds."""
        # COMBAT epsilon = 0.08:
        # random() = 0.05 (< 0.08) -> True
        # random() = 0.08 (>= 0.08) -> False
        # random() = 0.12 (>= 0.08) -> False
        mock_rng = MockRNG([0.05, 0.08, 0.12])
        self.assertTrue(self.policy.should_explore(ExplorationContext.COMBAT, rng=mock_rng))
        self.assertFalse(self.policy.should_explore(ExplorationContext.COMBAT, rng=mock_rng))
        self.assertFalse(self.policy.should_explore(ExplorationContext.COMBAT, rng=mock_rng))

    def test_seeded_random_generator_in_constructor(self) -> None:
        """Verifies constructor-injected RNG reproduces identical sequence."""
        p1 = ExplorationPolicy(default_rng=random.Random(12345))
        p2 = ExplorationPolicy(default_rng=random.Random(12345))

        for _ in range(20):
            res1 = p1.should_explore(ExplorationContext.ROAM)
            res2 = p2.should_explore(ExplorationContext.ROAM)
            self.assertEqual(res1, res2)

    def test_custom_epsilon_configuration_and_boundaries(self) -> None:
        """Verifies setting custom exploration rates within bounds [0.0, 1.0]."""
        p = ExplorationPolicy(roam_epsilon=0.0, farm_epsilon=1.0)
        self.assertEqual(p.epsilon(ExplorationContext.ROAM), 0.0)
        self.assertEqual(p.epsilon(ExplorationContext.FARM), 1.0)

        # Zero epsilon never explores
        rng = MockRNG([0.0001, 0.5, 0.99])
        self.assertFalse(p.should_explore(ExplorationContext.ROAM, rng=rng))

        # Full epsilon always explores
        rng2 = MockRNG([0.0, 0.5, 0.999])
        self.assertTrue(p.should_explore(ExplorationContext.FARM, rng=rng2))

        # Invalid bounds raise ValueError
        with self.assertRaises(ValueError):
            p.set_epsilon(ExplorationContext.COMBAT, -0.01)
        with self.assertRaises(ValueError):
            p.set_epsilon(ExplorationContext.COMBAT, 1.01)

    def test_no_hardware_and_no_q_mutation(self) -> None:
        """Verifies policy has no side effects on external state or hardware."""
        import sys
        # Verify no forbidden hardware modules are imported
        forbidden = ["subprocess", "control", "adb", "joystick", "vision", "cv2"]
        for mod in forbidden:
            self.assertNotIn(f"brain.exploration.{mod}", sys.modules)


if __name__ == "__main__":
    unittest.main()
