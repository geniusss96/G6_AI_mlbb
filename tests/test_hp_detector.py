"""
Unit tests for vision.hp_detector module.
Tests HPObservation dataclass, player HP detection, enemy HP hue filter,
red optical bar detector, and enemy minion filter on synthetic frames.
"""

import math
import numpy as np
import cv2

from config.config import SCREEN_WIDTH, SCREEN_HEIGHT
from vision.detector import Detection
from vision.hp_detector import HPObservation, HPDetector

TOLERANCE = 1e-3


def test_hp_observation_dataclass():
    """Verify HPObservation dataclass fields."""
    obs = HPObservation(value=0.75, confidence=1.0, visible=True)
    assert math.isclose(obs.value, 0.75, rel_tol=TOLERANCE)
    assert math.isclose(obs.confidence, 1.0, rel_tol=TOLERANCE)
    assert obs.visible is True


def test_detect_player_hp_empty_or_none():
    """None or missing target returns default healthy observation."""
    detector = HPDetector()
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    res1 = detector.detect_player_hp(None, (500, 500))
    assert res1.value == 1.0
    assert res1.visible is False

    res2 = detector.detect_player_hp(frame, None)
    assert res2.value == 1.0
    assert res2.visible is False


def test_detect_player_hp_synthetic():
    """Synthetic frame with green pixels above player center."""
    detector = HPDetector()
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    sx, sy = 772, 360

    # Green color in HSV: H=60, S=200, V=200
    green_hsv = np.uint8([[[60, 200, 200]]])
    green_bgr = cv2.cvtColor(green_hsv, cv2.COLOR_HSV2BGR)[0, 0]

    # Paint 60 green pixels in ROI (sy - 32 to sy + 2, sx - 45 to sx + 45)
    # Expected ratio = 60 / 120.0 = 0.50
    frame[sy - 20:sy - 10, sx - 3:sx + 3] = green_bgr

    obs = detector.detect_player_hp(frame, (sx, sy))
    assert obs.visible is True
    assert math.isclose(obs.value, 60 / 120.0, rel_tol=TOLERANCE)


def test_is_real_enemy_hp_bar():
    """Verify red HP bar verification against allies (green) and enemies (red)."""
    detector = HPDetector()
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    # 1. Paint red patch (enemy)
    red_hsv = np.uint8([[[5, 200, 200]]])
    red_bgr = cv2.cvtColor(red_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    frame[190:210, 290:310] = red_bgr

    assert detector.is_real_enemy_hp_bar(frame, [300, 200, 20, 20]) is True

    # 2. Paint green patch (ally)
    green_hsv = np.uint8([[[60, 200, 200]]])
    green_bgr = cv2.cvtColor(green_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    frame[190:210, 490:510] = green_bgr

    assert detector.is_real_enemy_hp_bar(frame, [500, 200, 20, 20]) is False


def test_detect_red_hp_bars_aspect_and_solidity():
    """Monolithic solid bar passes, crumpled or too square shapes fail."""
    detector = HPDetector()
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    red_hsv = np.uint8([[[5, 200, 200]]])
    red_bgr = cv2.cvtColor(red_hsv, cv2.COLOR_HSV2BGR)[0, 0]

    # Solid bar: 50px wide x 10px high (aspect 5.0, solidity 1.0) in safe combat zone (y=300, x=700)
    frame[300:310, 700:750] = red_bgr

    bars = detector.detect_red_hp_bars(frame)
    assert len(bars) == 1
    cx, cy, bw, bh = bars[0]
    assert math.isclose(bw, 50.0, abs_tol=1.0)
    assert math.isclose(bh, 10.0, abs_tol=1.0)


def test_enemy_minion_filter():
    """Allied minions with green bar or on base are rejected."""
    detector = HPDetector()
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    # Near base: always False
    assert detector.is_real_enemy_minion(frame, [500, 300, 40, 40], is_near_base=True) is False

    # Out of base with no green: True
    assert detector.is_real_enemy_minion(frame, [500, 300, 40, 40], is_near_base=False) is True


if __name__ == "__main__":
    test_hp_observation_dataclass()
    test_detect_player_hp_empty_or_none()
    test_detect_player_hp_synthetic()
    test_is_real_enemy_hp_bar()
    test_detect_red_hp_bars_aspect_and_solidity()
    test_enemy_minion_filter()
    print("All vision.hp_detector tests passed successfully!")
