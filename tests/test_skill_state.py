"""
Unit tests for vision.skill_state module.
Verifies SkillState dataclass and SkillStateChecker on synthetic pixel patches.
"""

import numpy as np
import cv2

from config.config import SCREEN_WIDTH, SCREEN_HEIGHT, SKILL_CHECK_COORDS
from vision.skill_state import SkillState, SkillStateChecker


def test_skill_state_dataclass():
    """Verify SkillState fields, to_dict(), dict indexing, and get()."""
    state = SkillState(s1_ready=True, s2_ready=False, ultimate_ready=True)

    assert state.s1_ready is True
    assert state.s2_ready is False
    assert state.ultimate_ready is True

    # Dictionary conversion
    d = state.to_dict()
    assert d == {"s1": True, "s2": False, "ult": True}

    # Dict-like access
    assert state["s1"] is True
    assert state["s2"] is False
    assert state["ult"] is True
    assert state["ultimate"] is True

    # Dict-like get()
    assert state.get("s1") is True
    assert state.get("s2") is False
    assert state.get("unknown", False) is False


def test_checker_all_not_ready():
    """Black frame (0 saturation, 0 value) -> all skills not ready."""
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    checker = SkillStateChecker()
    state = checker.check(frame)

    assert state.s1_ready is False
    assert state.s2_ready is False
    assert state.ultimate_ready is False


def test_checker_individual_skills_ready():
    """
    Synthetic frame where only specific skill button patches have
    high saturation (>130) and high value (>75).
    """
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    checker = SkillStateChecker()

    # BGR color for HSV (90, 200, 200) -> S=200 (>130), V=200 (>75)
    bright_hsv = np.uint8([[[90, 200, 200]]])
    bright_bgr = cv2.cvtColor(bright_hsv, cv2.COLOR_HSV2BGR)[0, 0]

    # Paint S1 button area only
    s1_cx, s1_cy = SKILL_CHECK_COORDS["s1"]
    frame[s1_cy - 8:s1_cy + 8, s1_cx - 8:s1_cx + 8] = bright_bgr

    state = checker.check(frame)
    assert state.s1_ready is True
    assert state.s2_ready is False
    assert state.ultimate_ready is False

    # Paint S2 button area as well
    s2_cx, s2_cy = SKILL_CHECK_COORDS["s2"]
    frame[s2_cy - 8:s2_cy + 8, s2_cx - 8:s2_cx + 8] = bright_bgr

    state = checker.check(frame)
    assert state.s1_ready is True
    assert state.s2_ready is True
    assert state.ultimate_ready is False

    # Paint Ult button area
    ult_cx, ult_cy = SKILL_CHECK_COORDS["ult"]
    frame[ult_cy - 8:ult_cy + 8, ult_cx - 8:ult_cx + 8] = bright_bgr

    state = checker.check(frame)
    assert state.s1_ready is True
    assert state.s2_ready is True
    assert state.ultimate_ready is True


def test_checker_empty_or_none_frame():
    """Graceful handling of None or empty frames."""
    checker = SkillStateChecker()
    assert checker.get_readiness(None) == {"s1": False, "s2": False, "ult": False}
    assert checker.check(None) == SkillState(False, False, False)


if __name__ == "__main__":
    test_skill_state_dataclass()
    test_checker_all_not_ready()
    test_checker_individual_skills_ready()
    test_checker_empty_or_none_frame()
    print("All vision.skill_state tests passed successfully!")
