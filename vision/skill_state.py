"""
vision.skill_state — Optical readiness detector for Claude's skills.

Analyzes HSV saturation and value channels at predetermined button ROIs
to determine if skills (S1, S2, Ultimate) are off cooldown.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import cv2
import numpy as np

from config.config import SKILL_CHECK_COORDS


@dataclass(frozen=True)
class SkillState:
    """
    Immutable representation of hero skill readiness.
    Pure CV descriptor without tactical cooldown logic.
    """
    s1_ready: bool
    s2_ready: bool
    ultimate_ready: bool

    def to_dict(self) -> Dict[str, bool]:
        """Backwards-compatible dictionary format for V1 consumers."""
        return {
            "s1": self.s1_ready,
            "s2": self.s2_ready,
            "ult": self.ultimate_ready,
        }

    def __getitem__(self, key: str) -> bool:
        """Allows dictionary-style access e.g. state['s1']."""
        if key == "s1":
            return self.s1_ready
        elif key == "s2":
            return self.s2_ready
        elif key in ("ult", "ultimate"):
            return self.ultimate_ready
        raise KeyError(f"Invalid skill name: {key}")

    def get(self, key: str, default: bool = False) -> bool:
        """Allows dict-like get() method e.g. state.get('s1', False)."""
        try:
            return self[key]
        except KeyError:
            return default


class SkillStateChecker:
    """
    Optical sensor for detecting skill readiness from HSV frame crops.
    """

    def __init__(self, skill_coords: Optional[Dict[str, Tuple[int, int]]] = None):
        self.skill_coords = skill_coords or SKILL_CHECK_COORDS

    def check(self, frame: np.ndarray) -> SkillState:
        """
        Calculates saturation and value means over 16x16 pixel patches
        around configured skill button centers.

        Args:
            frame: Reference resolution BGR image (1544x720).

        Returns:
            SkillState dataclass instance.
        """
        readiness_dict = self.get_readiness(frame)
        return SkillState(
            s1_ready=readiness_dict.get("s1", False),
            s2_ready=readiness_dict.get("s2", False),
            ultimate_ready=readiness_dict.get("ult", False),
        )

    def get_readiness(self, frame: np.ndarray) -> Dict[str, bool]:
        """
        Calculates raw readiness dictionary {'s1': bool, 's2': bool, 'ult': bool}.
        Preserves exact V1 HSV analysis logic.
        """
        if frame is None or frame.size == 0:
            return {"s1": False, "s2": False, "ult": False}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        status: Dict[str, bool] = {}

        for name, (cx, cy) in self.skill_coords.items():
            patch = hsv[
                max(0, cy - 8):min(frame.shape[0], cy + 8),
                max(0, cx - 8):min(frame.shape[1], cx + 8)
            ]
            if patch.size == 0:
                status[name] = False
                continue

            mean_s = np.mean(patch[:, :, 1])
            mean_v = np.mean(patch[:, :, 2])
            status[name] = bool((mean_s > 130) and (mean_v > 75))

        return status
