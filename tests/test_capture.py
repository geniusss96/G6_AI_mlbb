"""
Unit tests for vision.capture module.
Mock Win32 and mss calls to verify logic without real device or window.
"""

from unittest.mock import patch, MagicMock
import numpy as np

from config.config import SCREEN_WIDTH, SCREEN_HEIGHT
from vision.capture import ScreenCapture


def test_init_and_config():
    """Verify ScreenCapture initializes with default keywords and no hardcoded dimensions."""
    cap = ScreenCapture()
    assert "scrcpy" in cap.window_title_keywords
    assert "sm-" in cap.window_title_keywords
    assert cap._sct is None


def test_restore_if_minimized():
    """Verify window restoration when minimized (coordinates < -1000)."""
    cap = ScreenCapture()
    mock_hwnd = 12345

    with patch("vision.capture.win32gui.GetWindowRect") as mock_get_rect, \
         patch("vision.capture.win32gui.ShowWindow") as mock_show:
        
        # 1. Window is minimized (-32000, -32000), then restored to (100, 100)
        mock_get_rect.side_effect = [
            (-32000, -32000, -31000, -31000),
            (100, 100, 1370, 692)
        ]

        result_hwnd = cap.restore_if_minimized(mock_hwnd)
        assert result_hwnd == mock_hwnd
        mock_show.assert_called_once()


def test_restore_not_needed():
    """Verify normal windows are not unnecessarily restored."""
    cap = ScreenCapture()
    mock_hwnd = 12345

    with patch("vision.capture.win32gui.GetWindowRect") as mock_get_rect, \
         patch("vision.capture.win32gui.ShowWindow") as mock_show:
        
        mock_get_rect.return_value = (100, 100, 1370, 692)
        result_hwnd = cap.restore_if_minimized(mock_hwnd)
        assert result_hwnd == mock_hwnd
        mock_show.assert_not_called()


def test_grab_normalizes_to_canonical():
    """Verify grab passes raw frame through geometry.normalize_frame to SCREEN_WIDTH x SCREEN_HEIGHT."""
    cap = ScreenCapture()
    mock_hwnd = 12345

    # Create dummy frame at 1270x592 BGRA (4 channels as mss produces)
    fake_bgra = np.zeros((592, 1270, 4), dtype=np.uint8)

    with patch.object(cap, "find_scrcpy_window", return_value=mock_hwnd), \
         patch.object(cap, "get_client_area", return_value={"top": 0, "left": 0, "width": 1270, "height": 592}), \
         patch.object(cap, "_ensure_sct") as mock_sct:
        
        mock_sct.return_value.grab.return_value = fake_bgra

        frame = cap.grab()
        assert frame is not None
        assert frame.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)


if __name__ == "__main__":
    test_init_and_config()
    test_restore_if_minimized()
    test_restore_not_needed()
    test_grab_normalizes_to_canonical()
    print("All vision.capture tests passed successfully!")
