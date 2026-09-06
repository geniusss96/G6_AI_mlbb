"""
vision.capture — Screen capture service for Scrcpy/Android display.

Extracts window location, client area bounding, minimized window restoration,
and frame grab via mss with canonical geometry normalization.
"""

from typing import Optional, Dict, Any
import time
import cv2
import numpy as np
import mss
import win32gui
import win32con

from vision.geometry import normalize_frame


class ScreenCapture:
    """
    Manages capturing video frames from the Scrcpy window on Windows using mss.
    Automatically finds Scrcpy window, restores it if minimized, and returns
    normalized frames (1544x720).
    """

    def __init__(self, window_title_keywords: Optional[list[str]] = None):
        self.window_title_keywords = window_title_keywords or ["scrcpy", "sm-"]
        self._sct: Optional[mss.mss] = None

    def __enter__(self):
        self._sct = mss.mss()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def _ensure_sct(self) -> mss.mss:
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def find_scrcpy_window(self) -> Optional[int]:
        """
        Locates the Scrcpy window handle (HWND) via class name 'SDL_app'
        or window titles matching keywords. Automatically restores if minimized.
        """
        hwnd = win32gui.FindWindow("SDL_app", None)
        if not hwnd:
            def enum_cb(h, acc):
                if win32gui.IsWindowVisible(h):
                    txt = win32gui.GetWindowText(h)
                    cls = win32gui.GetClassName(h)
                    if cls == "SDL_app" or any(kw in txt.lower() for kw in self.window_title_keywords):
                        acc.append(h)
            acc = []
            win32gui.EnumWindows(enum_cb, acc)
            if acc:
                hwnd = acc[0]

        if hwnd and win32gui.IsWindowVisible(hwnd):
            hwnd = self.restore_if_minimized(hwnd)
            return hwnd
        return None

    def restore_if_minimized(self, hwnd: int) -> Optional[int]:
        """
        Checks if window is minimized (coordinates < -1000) and restores it with SW_RESTORE.
        """
        try:
            rect = win32gui.GetWindowRect(hwnd)
            if rect[0] < -1000 or rect[1] < -1000:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.1)
                rect = win32gui.GetWindowRect(hwnd)
            if rect[0] > -1000 and rect[1] > -1000:
                return hwnd
        except Exception:
            return None
        return None

    def get_client_area(self, hwnd: int) -> Optional[Dict[str, int]]:
        """
        Calculates the screen region for mss grab from window client area.
        """
        try:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width, height = right - left, bottom - top
            if width <= 0 or height <= 0:
                return None
            screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
            return {
                "top": max(0, screen_top),
                "left": max(0, screen_left),
                "width": width,
                "height": height
            }
        except Exception:
            return None

    def grab(self, hwnd: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Captures one frame from the Scrcpy window.
        Returns BGR numpy array normalized to canonical reference resolution (1544x720),
        or None if capture failed.
        """
        if hwnd is None:
            hwnd = self.find_scrcpy_window()
        if not hwnd:
            return None

        monitor = self.get_client_area(hwnd)
        if not monitor or monitor["width"] <= 0 or monitor["height"] <= 0:
            return None

        sct = self._ensure_sct()
        sct_img = sct.grab(monitor)
        raw_frame = np.array(sct_img)
        bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
        bgr_frame = np.ascontiguousarray(bgr_frame)

        # Normalize resolution using canonical geometry module (1544x720)
        return normalize_frame(bgr_frame)
