"""
Unit tests for vision.geometry module.
"""

import math
import numpy as np

from config.config import SCREEN_WIDTH, SCREEN_HEIGHT
from vision.geometry import (
    normalize_point,
    denormalize_point,
    scale_rect,
    normalize_frame,
)

TOLERANCE = 1e-4


def test_normalize_identity():
    """Points already in reference coordinates (1544x720) should not change."""
    x, y = 772.0, 360.0
    nx, ny = normalize_point(x, y, SCREEN_WIDTH, SCREEN_HEIGHT)
    assert math.isclose(nx, x, rel_tol=TOLERANCE)
    assert math.isclose(ny, y, rel_tol=TOLERANCE)


def test_normalize_from_scrcpy_resolution():
    """1270x592 source coordinates map correctly to 1544x720."""
    src_w, src_h = 1270, 592
    
    # Origin
    ox, oy = normalize_point(0, 0, src_w, src_h)
    assert math.isclose(ox, 0.0, abs_tol=TOLERANCE)
    assert math.isclose(oy, 0.0, abs_tol=TOLERANCE)

    # Max bound
    mx, my = normalize_point(1270, 592, src_w, src_h)
    assert math.isclose(mx, float(SCREEN_WIDTH), rel_tol=TOLERANCE)
    assert math.isclose(my, float(SCREEN_HEIGHT), rel_tol=TOLERANCE)

    # Center
    cx, cy = normalize_point(635, 296, src_w, src_h)
    assert math.isclose(cx, SCREEN_WIDTH / 2.0, rel_tol=TOLERANCE)
    assert math.isclose(cy, SCREEN_HEIGHT / 2.0, rel_tol=TOLERANCE)


def test_roundtrip_point():
    """Converting from source -> ref -> source preserves original point."""
    src_w, src_h = 1270, 592
    orig_x, orig_y = 315.4, 482.1

    ref_x, ref_y = normalize_point(orig_x, orig_y, src_w, src_h)
    recovered_x, recovered_y = denormalize_point(ref_x, ref_y, src_w, src_h)

    assert math.isclose(recovered_x, orig_x, rel_tol=TOLERANCE)
    assert math.isclose(recovered_y, orig_y, rel_tol=TOLERANCE)


def test_scale_rect():
    """Bounding box scales properly into reference frame."""
    src_w, src_h = 1270, 592
    x1, y1, x2, y2 = 100.0, 50.0, 200.0, 150.0

    rx1, ry1, rx2, ry2 = scale_rect(x1, y1, x2, y2, src_w, src_h)
    expected_x1, expected_y1 = normalize_point(x1, y1, src_w, src_h)
    expected_x2, expected_y2 = normalize_point(x2, y2, src_w, src_h)

    assert math.isclose(rx1, expected_x1, rel_tol=TOLERANCE)
    assert math.isclose(ry1, expected_y1, rel_tol=TOLERANCE)
    assert math.isclose(rx2, expected_x2, rel_tol=TOLERANCE)
    assert math.isclose(ry2, expected_y2, rel_tol=TOLERANCE)


def test_normalize_frame_same_size():
    """Frames already at 1544x720 are returned unchanged (identity)."""
    frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    normalized = normalize_frame(frame)
    assert normalized is frame  # Identical reference
    assert normalized.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)


def test_normalize_frame_resize():
    """Frames of other sizes (e.g., 1270x592) get resized to 1544x720."""
    frame = np.zeros((592, 1270, 3), dtype=np.uint8)
    normalized = normalize_frame(frame)
    assert normalized.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)


if __name__ == "__main__":
    test_normalize_identity()
    test_normalize_from_scrcpy_resolution()
    test_roundtrip_point()
    test_scale_rect()
    test_normalize_frame_same_size()
    test_normalize_frame_resize()
    print("All geometry tests passed successfully!")
