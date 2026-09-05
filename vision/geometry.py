"""
vision.geometry — Reference coordinate system and resolution transformations.

Internal canonical reference coordinate system: SCREEN_WIDTH x SCREEN_HEIGHT (1544 x 720).
All coordinates inside WorldState, Vision, Brain, and Joystick map through this module.
"""

from typing import Tuple
import cv2
import numpy as np

from config.config import SCREEN_WIDTH, SCREEN_HEIGHT


def normalize_point(
    x: float,
    y: float,
    source_width: int,
    source_height: int,
) -> Tuple[float, float]:
    """
    Transforms a 2D coordinate from an arbitrary source resolution
    into the reference coordinate system (SCREEN_WIDTH x SCREEN_HEIGHT).

    Args:
        x: X-coordinate in source resolution.
        y: Y-coordinate in source resolution.
        source_width: Width of the source frame/window.
        source_height: Height of the source frame/window.

    Returns:
        (x_ref, y_ref): Scaled coordinates in reference resolution.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid source dimensions: {source_width}x{source_height}")

    scale_x = SCREEN_WIDTH / float(source_width)
    scale_y = SCREEN_HEIGHT / float(source_height)

    return (x * scale_x, y * scale_y)


def denormalize_point(
    x: float,
    y: float,
    target_width: int,
    target_height: int,
) -> Tuple[float, float]:
    """
    Transforms a 2D coordinate from reference coordinates (SCREEN_WIDTH x SCREEN_HEIGHT)
    back to a target resolution.

    Args:
        x: X-coordinate in reference resolution.
        y: Y-coordinate in reference resolution.
        target_width: Width of the target resolution.
        target_height: Height of the target resolution.

    Returns:
        (x_target, y_target): Scaled coordinates in target resolution.
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target dimensions: {target_width}x{target_height}")

    scale_x = float(target_width) / SCREEN_WIDTH
    scale_y = float(target_height) / SCREEN_HEIGHT

    return (x * scale_x, y * scale_y)


def scale_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    source_width: int,
    source_height: int,
) -> Tuple[float, float, float, float]:
    """
    Transforms bounding box coordinates (x1, y1, x2, y2) from an arbitrary source
    resolution into the reference resolution (SCREEN_WIDTH x SCREEN_HEIGHT).

    Args:
        x1, y1: Top-left coordinate in source resolution.
        x2, y2: Bottom-right coordinate in source resolution.
        source_width: Width of the source frame/window.
        source_height: Height of the source frame/window.

    Returns:
        (x1_ref, y1_ref, x2_ref, y2_ref): Scaled rectangle in reference resolution.
    """
    nx1, ny1 = normalize_point(x1, y1, source_width, source_height)
    nx2, ny2 = normalize_point(x2, y2, source_width, source_height)
    return (nx1, ny1, nx2, ny2)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Ensures the image frame is scaled to the canonical reference resolution
    (SCREEN_WIDTH x SCREEN_HEIGHT).

    If the incoming frame is already of shape (SCREEN_HEIGHT, SCREEN_WIDTH),
    it is returned directly without copying or resizing.

    Args:
        frame: Input image as a numpy array (BGR or RGB).

    Returns:
        Normalized frame of size (SCREEN_HEIGHT, SCREEN_WIDTH).
    """
    if frame is None:
        raise ValueError("Cannot normalize a None frame.")

    h, w = frame.shape[:2]
    if w == SCREEN_WIDTH and h == SCREEN_HEIGHT:
        return frame

    return cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_LINEAR)
