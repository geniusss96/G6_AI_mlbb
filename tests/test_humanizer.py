"""
Unit tests for control.humanizer (InputHumanizer).
Verifies spatial jitter clamping, hold duration bounds, deterministic RNG,
NaN/Inf handling, independence from ADB, and absence of time.sleep calls.
"""

import os
import sys
import math
import random
import unittest
from unittest.mock import patch

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from config.config import (
    SPATIAL_JITTER_SIGMA,
    SPATIAL_MAX_OFFSET,
    HOLD_DURATION_MEAN_MS,
    HOLD_DURATION_MIN_MS,
    HOLD_DURATION_MAX_MS,
)
from control.humanizer import InputHumanizer


class TestInputHumanizer(unittest.TestCase):
    def setUp(self):
        self.humanizer = InputHumanizer()

    def test_spatial_jitter_clamping_and_bounds(self):
        """Verify jitter offsets never exceed max_offset across many samples."""
        base_x, base_y = 500, 300
        max_offset = SPATIAL_MAX_OFFSET

        for _ in range(1000):
            jx, jy = self.humanizer.jitter_point(base_x, base_y)
            dx = abs(jx - base_x)
            dy = abs(jy - base_y)
            self.assertLessEqual(dx, max_offset)
            self.assertLessEqual(dy, max_offset)
            self.assertIsInstance(jx, int)
            self.assertIsInstance(jy, int)

    def test_custom_spatial_jitter_params(self):
        """Verify custom sigma and max_offset are respected."""
        base_x, base_y = 100, 200
        custom_max = 2

        for _ in range(500):
            jx, jy = self.humanizer.jitter_point(base_x, base_y, std_dev=1.0, max_offset=custom_max)
            self.assertLessEqual(abs(jx - base_x), custom_max)
            self.assertLessEqual(abs(jy - base_y), custom_max)

    def test_hold_duration_bounds_and_positivity(self):
        """Verify hold duration is strictly clamped between MIN and MAX and non-negative."""
        for _ in range(1000):
            duration = self.humanizer.hold_duration_ms()
            self.assertGreaterEqual(duration, int(HOLD_DURATION_MIN_MS))
            self.assertLessEqual(duration, int(HOLD_DURATION_MAX_MS))
            self.assertGreater(duration, 0)
            self.assertIsInstance(duration, int)

    def test_custom_hold_duration_bounds(self):
        """Verify custom hold duration parameters."""
        for _ in range(500):
            duration = self.humanizer.hold_duration_ms(
                mean_ms=50.0, std_ms=5.0, min_ms=40.0, max_ms=60.0
            )
            self.assertGreaterEqual(duration, 40)
            self.assertLessEqual(duration, 60)

    def test_reaction_delay_bounds(self):
        """Verify reaction delay obeys minimum floor."""
        for _ in range(500):
            delay = self.humanizer.reaction_delay_ms(base_ms=30.0, variance_ms=5.0, min_floor_ms=8.0)
            self.assertGreaterEqual(delay, 8)
            self.assertIsInstance(delay, int)

    def test_deterministic_behavior_with_seeded_rng(self):
        """Two humanizers with identical seeded RNGs produce identical sequences."""
        rng1 = random.Random(1337)
        rng2 = random.Random(1337)

        h1 = InputHumanizer(rng=rng1)
        h2 = InputHumanizer(rng=rng2)

        for _ in range(50):
            p1 = h1.jitter_point(640, 360)
            p2 = h2.jitter_point(640, 360)
            self.assertEqual(p1, p2)

            d1 = h1.hold_duration_ms()
            d2 = h2.hold_duration_ms()
            self.assertEqual(d1, d2)

            r1 = h1.reaction_delay_ms()
            r2 = h2.reaction_delay_ms()
            self.assertEqual(r1, r2)

    def test_rejection_of_nan_and_inf(self):
        """Invalid coordinate values must raise ValueError."""
        with self.assertRaises(ValueError):
            self.humanizer.jitter_point(float("nan"), 100.0)

        with self.assertRaises(ValueError):
            self.humanizer.jitter_point(100.0, float("inf"))

        with self.assertRaises(ValueError):
            self.humanizer.jitter_point(float("-inf"), 100.0)

        with self.assertRaises(ValueError):
            InputHumanizer.jitter_coordinate(float("nan"), 100)

    def test_v1_static_method_compatibility(self):
        """Verify V1 static method calls work identically."""
        jx, jy = InputHumanizer.jitter_coordinate(500, 300)
        self.assertLessEqual(abs(jx - 500), SPATIAL_MAX_OFFSET)
        self.assertLessEqual(abs(jy - 300), SPATIAL_MAX_OFFSET)

        hold_ms = InputHumanizer.get_touch_hold_duration()
        self.assertGreaterEqual(hold_ms, int(HOLD_DURATION_MIN_MS))
        self.assertLessEqual(hold_ms, int(HOLD_DURATION_MAX_MS))

        delay_ms = InputHumanizer.get_reaction_delay(base_delay_ms=25.0)
        self.assertGreaterEqual(delay_ms, 8)

    def test_no_sleep_called(self):
        """Humanizer must only compute values and never call time.sleep."""
        with patch("time.sleep") as mock_sleep:
            self.humanizer.jitter_point(100, 200)
            self.humanizer.hold_duration_ms()
            self.humanizer.reaction_delay_ms()
            mock_sleep.assert_not_called()

    def test_no_adb_dependency(self):
        """Humanizer must not import ADB or depend on control.adb."""
        import control.humanizer as h_module
        self.assertFalse(hasattr(h_module, "ADBTransport"))
        self.assertFalse(hasattr(h_module, "subprocess"))


if __name__ == "__main__":
    unittest.main()
