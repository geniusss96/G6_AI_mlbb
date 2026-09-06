"""
vision.hp_detector — Optical HP bar sensor and color verification.

Isolates optical HP sensing for player Claude and enemy hero bars/minions
without business logic, tracking, death detection, or combat decisions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from vision.detector import Detection


@dataclass(frozen=True)
class HPObservation:
    """
    Structured computer vision observation of an entity's health bar.
    """
    value: float        # Normalized ratio (0.0 .. 1.0)
    confidence: float   # 1.0 when clearly visible/verified, 0.0 when missing/fallback
    visible: bool       # True if green/red bar is detected


class HPDetector:
    """
    Optical sensor for detecting and verifying HP bars in MLBB frames (1544x720).
    """

    def detect_player_hp(
        self,
        frame: np.ndarray,
        self_target: Optional[Union[Tuple[float, float], List[float], Detection]] = None
    ) -> HPObservation:
        """
        Estimates player Claude's HP percentage (0.0 .. 1.0).
        Analyzes green HP bar above hero character model.

        Args:
            frame: Reference resolution frame (1544x720 BGR).
            self_target: Hero center coordinate (x, y) or Detection object.

        Returns:
            HPObservation with value in [0.1, 1.0].
        """
        if frame is None or frame.size == 0 or not self_target:
            return HPObservation(value=1.0, confidence=0.0, visible=False)

        if isinstance(self_target, Detection):
            sx, sy = int(self_target.center[0]), int(self_target.center[1])
        elif isinstance(self_target, (list, tuple)):
            sx, sy = int(self_target[0]), int(self_target[1])
        else:
            return HPObservation(value=1.0, confidence=0.0, visible=False)

        h, w = frame.shape[:2]
        y1, y2 = max(0, sy - 32), min(h, sy + 2)
        x1, x2 = max(0, sx - 45), min(w, sx + 45)
        patch = frame[y1:y2, x1:x2]

        if patch.size == 0:
            return HPObservation(value=1.0, confidence=0.0, visible=False)

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 75, 75), (85, 255, 255))
        green_pixels = np.count_nonzero(green_mask)

        hp_ratio = float(np.clip(green_pixels / 120.0, 0.1, 1.0))
        visible = green_pixels > 12

        return HPObservation(
            value=hp_ratio,
            confidence=1.0 if visible else 0.5,
            visible=visible
        )

    def is_real_enemy_hp_bar(
        self,
        frame: np.ndarray,
        xywh_or_detection: Union[List[float], Tuple[float, float, float, float], Detection]
    ) -> bool:
        """
        100% Enemy Hue Verification filter:
        In MLBB, enemy heroes have strictly red HP bars (red_ratio >= 0.10).
        Allies/teammates have green or blue bars.

        Returns:
            True if patch is genuine enemy red bar, False otherwise.
        """
        if frame is None or frame.size == 0:
            return False

        if isinstance(xywh_or_detection, Detection):
            cx, cy = xywh_or_detection.center
            bw, bh = xywh_or_detection.width, xywh_or_detection.height
        else:
            cx, cy, bw, bh = xywh_or_detection

        h, w = frame.shape[:2]
        x1 = max(0, int(cx - bw / 2.0))
        x2 = min(w, int(cx + bw / 2.0))
        y1 = max(0, int(cy - bh / 2.0))
        y2 = min(h, int(cy + bh / 2.0))

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 4:
            return False

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 75, 75), (12, 255, 255))
        mask2 = cv2.inRange(hsv, (168, 75, 75), (180, 255, 255))
        red_pixels = np.count_nonzero(mask1 | mask2)
        total_pixels = patch.shape[0] * patch.shape[1]
        red_ratio = red_pixels / float(total_pixels)

        green_mask = cv2.inRange(hsv, (35, 70, 70), (85, 255, 255))
        green_pixels = np.count_nonzero(green_mask)
        green_ratio = green_pixels / float(total_pixels)

        if red_ratio < 0.10 or green_ratio > red_ratio:
            return False

        return True

    def detect_red_hp_bars(self, frame: np.ndarray) -> List[List[float]]:
        """
        Optical sensor for enemy HP bars with solidity and aspect ratio filtering.
        Eliminates crumpled red cloth of defeated heroes (solidity < 0.70)
        and preserves genuine rectangular health bars.

        Returns:
            List of [cx, cy, bw, bh] bounding boxes in reference resolution (1544x720).
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, (0, 110, 110), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 110, 110), (180, 255, 255))
        mask = mask1 | mask2

        # HUD exclusions:
        mask[:int(h * 0.18), :] = 0
        mask[int(h * 0.65):, int(w * 0.78):] = 0
        mask[:int(h * 0.40), :int(w * 0.22)] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[List[float]] = []

        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            # HP bar dimensions: width 25..200px, thin height 4..16px
            if 25 <= bw <= 200 and 4 <= bh <= 16:
                aspect = bw / float(bh)
                area = cv2.contourArea(c)
                solidity = area / float(bw * bh)
                # Monolithic health bar: solidity >= 0.70 and aspect >= 2.8
                if aspect >= 2.8 and solidity >= 0.70:
                    cx = bx + bw / 2.0
                    cy = by + bh / 2.0
                    boxes.append([cx, cy, float(bw), float(bh)])

        return boxes

    def is_real_enemy_minion(
        self,
        frame: np.ndarray,
        xywh_or_detection: Union[List[float], Tuple[float, float, float, float], Detection],
        is_near_base: bool
    ) -> bool:
        """
        Filters out allied minions near base or minions with green HP bar overhead.
        """
        if is_near_base or frame is None or frame.size == 0:
            return False

        if isinstance(xywh_or_detection, Detection):
            cx, cy = xywh_or_detection.center
            bw, bh = xywh_or_detection.width, xywh_or_detection.height
        else:
            cx, cy, bw, bh = xywh_or_detection

        h, w = frame.shape[:2]
        y1 = max(0, int(cy - bh / 2.0))
        y2 = min(h, int(cy - bh / 6.0))
        x1 = max(0, int(cx - bw / 2.0))
        x2 = min(w, int(cx + bw / 2.0))

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return False

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 70, 70), (85, 255, 255))
        green_ratio = np.count_nonzero(green_mask) / float(patch.shape[0] * patch.shape[1])

        if green_ratio > 0.15:
            return False

        return True
