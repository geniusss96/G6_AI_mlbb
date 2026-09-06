"""
control.humanizer — Input Humanization and Anti-Detection Engine for Claude Tactical AI V2.

Simulates realistic human physical touch biomechanics and reaction timing:
- 2D Gaussian spatial jitter around nominal coordinates
- Capacitive glass hold duration windows (45ms to 95ms)
- Reaction delay jitter between consecutive skill activations

Coordinate and hardware neutral: accepts raw numeric coordinates and returns jittered integers.
Pure mathematical/statistical module: no sleep, no threads, no ADB dependencies.
"""

from typing import Tuple, Optional, Any
import math
import random

from config.config import (
    SPATIAL_JITTER_SIGMA,
    SPATIAL_MAX_OFFSET,
    HOLD_DURATION_MEAN_MS,
    HOLD_DURATION_MIN_MS,
    HOLD_DURATION_MAX_MS,
)


class InputHumanizer:
    """
    Simulates realistic human physical touch characteristics.
    Supports optional RNG dependency injection for deterministic testing.
    """

    def __init__(
        self,
        spatial_sigma: float = SPATIAL_JITTER_SIGMA,
        max_offset: int = SPATIAL_MAX_OFFSET,
        hold_mean_ms: float = HOLD_DURATION_MEAN_MS,
        hold_min_ms: float = HOLD_DURATION_MIN_MS,
        hold_max_ms: float = HOLD_DURATION_MAX_MS,
        hold_std_ms: float = 10.0,
        rng: Optional[Any] = None,
    ):
        self.spatial_sigma = float(spatial_sigma)
        self.max_offset = int(max_offset)
        self.hold_mean_ms = float(hold_mean_ms)
        self.hold_min_ms = float(hold_min_ms)
        self.hold_max_ms = float(hold_max_ms)
        self.hold_std_ms = float(hold_std_ms)
        self.rng = rng if rng is not None else random

    def jitter_point(
        self,
        x: float,
        y: float,
        std_dev: Optional[float] = None,
        max_offset: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Applies 2D Gaussian jitter to target coordinates.
        Clamps offsets to prevent tapping outside target boundaries.

        Returns:
            (jittered_x, jittered_y) as integer screen pixels.
        """
        if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
            raise ValueError(f"Coordinates cannot be NaN or Inf: ({x}, {y})")

        sigma = std_dev if std_dev is not None else self.spatial_sigma
        limit = max_offset if max_offset is not None else self.max_offset

        dx = self.rng.gauss(0, sigma)
        dy = self.rng.gauss(0, sigma)
        dx = max(-limit, min(limit, dx))
        dy = max(-limit, min(limit, dy))

        return int(round(x + dx)), int(round(y + dy))

    def hold_duration_ms(
        self,
        mean_ms: Optional[float] = None,
        std_ms: Optional[float] = None,
        min_ms: Optional[float] = None,
        max_ms: Optional[float] = None,
    ) -> int:
        """
        Calculates physical glass contact duration in milliseconds.
        Guaranteed to be clamped between min_ms and max_ms (non-negative).
        """
        m = mean_ms if mean_ms is not None else self.hold_mean_ms
        s = std_ms if std_ms is not None else self.hold_std_ms
        mn = min_ms if min_ms is not None else self.hold_min_ms
        mx = max_ms if max_ms is not None else self.hold_max_ms

        val = self.rng.gauss(m, s)
        clamped = max(mn, min(mx, val))
        return int(round(clamped))

    def reaction_delay_ms(
        self,
        base_ms: float = 30.0,
        variance_ms: float = 5.0,
        min_floor_ms: float = 8.0,
        **kwargs,
    ) -> int:
        """
        Calculates inter-action reaction delay with Gaussian variance.
        Guaranteed to be >= min_floor_ms.
        """
        if "base_delay_ms" in kwargs:
            base_ms = kwargs["base_delay_ms"]
        delay = self.rng.gauss(base_ms, variance_ms)
        clamped = max(min_floor_ms, delay)
        return int(round(clamped))

    # --- Static V1-compatible interface ---

    @staticmethod
    def jitter_coordinate(
        x: int,
        y: int,
        std_dev: float = SPATIAL_JITTER_SIGMA,
        max_offset: int = SPATIAL_MAX_OFFSET,
    ) -> Tuple[int, int]:
        """V1-compatible static method."""
        if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
            raise ValueError(f"Coordinates cannot be NaN or Inf: ({x}, {y})")
        dx = random.gauss(0, std_dev)
        dy = random.gauss(0, std_dev)
        dx = max(-max_offset, min(max_offset, dx))
        dy = max(-max_offset, min(max_offset, dy))
        return int(round(x + dx)), int(round(y + dy))

    @staticmethod
    def get_touch_hold_duration(
        mean_ms: float = HOLD_DURATION_MEAN_MS,
        std_ms: float = 10.0,
        min_ms: float = HOLD_DURATION_MIN_MS,
        max_ms: float = HOLD_DURATION_MAX_MS,
    ) -> int:
        """V1-compatible static method."""
        val = random.gauss(mean_ms, std_ms)
        clamped = max(min_ms, min(max_ms, val))
        return int(round(clamped))

    @staticmethod
    def get_reaction_delay(
        base_ms: float = 30.0,
        variance_ms: float = 5.0,
        min_floor_ms: float = 8.0,
        **kwargs,
    ) -> int:
        """V1-compatible static method."""
        if "base_delay_ms" in kwargs:
            base_ms = kwargs["base_delay_ms"]
        delay = random.gauss(base_ms, variance_ms)
        clamped = max(min_floor_ms, delay)
        return int(round(clamped))
