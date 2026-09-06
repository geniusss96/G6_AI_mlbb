"""
Input Humanization & Anti-Detection Module
------------------------------------------
Provides realistic human physiological simulation for touch inputs:
  - Spatial Jitter: 2D Gaussian distribution around target center
  - Temporal Jitter: Normal distribution reaction delays (clipped to realistic bounds)
  - Natural Hold Windows: Touch-down to touch-up duration (40ms - 110ms)
  - Curved Touch Dragging: Cubic Bezier curves for directional skill targeting
"""

import math
import random
import time
from typing import List, Tuple


class InputHumanizer:
    """Simulates realistic human physical touch characteristics."""

    @staticmethod
    def jitter_coordinate(
        x: int,
        y: int,
        std_dev: float = 3.5,
        max_offset: int = 8
    ) -> Tuple[int, int]:
        """
        Applies 2D Gaussian jitter to a target coordinate.
        Human taps naturally cluster around the target center with a normal distribution.

        Args:
            x: Nominal X coordinate
            y: Nominal Y coordinate
            std_dev: Standard deviation of touch spread
            max_offset: Hard clamp to prevent tapping outside button bounds

        Returns:
            (jittered_x, jittered_y)
        """
        # Box-Muller / normal distribution offset
        dx = random.gauss(0, std_dev)
        dy = random.gauss(0, std_dev)

        # Clamp offsets within acceptable button radius
        dx = max(-max_offset, min(max_offset, dx))
        dy = max(-max_offset, min(max_offset, dy))

        return int(round(x + dx)), int(round(y + dy))

    @staticmethod
    def get_touch_hold_duration(
        mean_ms: float = 65.0,
        std_ms: float = 15.0,
        min_ms: float = 35.0,
        max_ms: float = 120.0
    ) -> float:
        """
        Returns realistic touch-down duration (seconds) before touch-up.
        Physical fingers contact capacitive glass for ~40-100ms.
        """
        duration_ms = random.gauss(mean_ms, std_ms)
        clamped_ms = max(min_ms, min(max_ms, duration_ms))
        return clamped_ms / 1000.0

    @staticmethod
    def get_reaction_delay(
        base_delay_ms: float,
        variance_ms: float = 20.0,
        min_floor_ms: float = 10.0
    ) -> float:
        """
        Calculates inter-action delay with Gaussian variance.
        Avoids exact rigid robotic intervals (e.g. exactly 50.00ms).
        """
        delay_ms = random.gauss(base_delay_ms, variance_ms)
        clamped_ms = max(min_floor_ms, delay_ms)
        return clamped_ms / 1000.0

    @staticmethod
    def generate_bezier_swipe(
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
        steps: int = 12,
        deviation: float = 25.0
    ) -> List[Tuple[int, int]]:
        """
        Generates a cubic Bezier curve simulating a natural curved swipe
        (e.g., aiming a directional skill or flicking the joystick).

        Human hand biomechanics naturally pivot around the thumb joint,
        producing a slight arc rather than an unnatural straight line.
        """
        x0, y0 = start_pt
        x3, y3 = end_pt

        # Midpoint vector
        mx = (x0 + x3) / 2.0
        my = (y0 + y3) / 2.0

        # Normal vector for lateral curvature
        dx = x3 - x0
        dy = y3 - y0
        dist = math.hypot(dx, dy) + 1e-5
        norm_x = -dy / dist
        norm_y = dx / dist

        # Random control point displacement
        curve_factor = random.uniform(-deviation, deviation)
        x1 = x0 + dx * 0.33 + norm_x * curve_factor
        y1 = y0 + dy * 0.33 + norm_y * curve_factor
        x2 = x0 + dx * 0.66 + norm_x * (curve_factor * 0.8)
        y2 = y0 + dy * 0.66 + norm_y * (curve_factor * 0.8)

        points = []
        for i in range(steps + 1):
            t = i / float(steps)
            u = 1.0 - t
            # Cubic Bezier formula: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
            bx = (u**3)*x0 + 3*(u**2)*t*x1 + 3*u*(t**2)*x2 + (t**3)*x3
            by = (u**3)*y0 + 3*(u**2)*t*y1 + 3*u*(t**2)*y2 + (t**3)*y3
            points.append((int(round(bx)), int(round(by))))

        return points
