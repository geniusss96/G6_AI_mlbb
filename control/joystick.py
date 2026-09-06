"""
control.joystick — Authoritative V2 Joystick Controller implementing JoystickPort.

Translates high-level movement and kiting vectors into touch screen motionevent sequences
via centralized ADBTransport.
Preserves exact V1 semantics:
- input motionevent DOWN / MOVE / UP
- Screen touch watchdog: notify_screen_touched() resets is_holding flag
- Automatic 1.8s watchdog refresh to prevent touch driver sleep
- 1px alternating spatial toggle to maintain movement stream
"""

import math
import time
from typing import Tuple, Optional, Callable

from control.interfaces import JoystickPort
from control.adb import ADBTransport
from config.config import (
    JOYSTICK_CENTER_X,
    JOYSTICK_CENTER_Y,
    JOYSTICK_RADIUS,
)


class JoystickControl(JoystickPort):
    """
    V2 Joystick Controller using centralized ADBTransport.
    Maintains continuous movement and kiting state.
    """

    def __init__(
        self,
        transport: ADBTransport,
        center_x: int = JOYSTICK_CENTER_X,
        center_y: int = JOYSTICK_CENTER_Y,
        radius: int = JOYSTICK_RADIUS,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.transport = transport
        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.radius = int(radius)
        self.time_provider = time_provider or time.time

        self.is_holding: bool = False
        self.current_pos: Tuple[int, int] = (self.center_x, self.center_y)
        self.last_down_time: float = 0.0
        self._toggle: bool = False

    def notify_screen_touched(self) -> None:
        """
        Resets holding state when another screen tap/swipe occurs.
        Forces the next movement command to send DOWN, preventing touch paralysis on Android.
        """
        self.is_holding = False

    def move_towards(
        self,
        hero_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        duration_ms: int = 400,
    ) -> Tuple[int, int]:
        """
        Directs the virtual joystick towards target_pos from hero_pos.
        Maintains touch contact with automatic re-DOWN watchdog.
        """
        hx, hy = hero_pos
        tx, ty = target_pos

        dx = tx - hx
        dy = ty - hy
        angle = math.atan2(dy, dx)

        target_joy_x = int(round(self.center_x + self.radius * math.cos(angle)))
        target_joy_y = int(round(self.center_y + self.radius * math.sin(angle)))

        self._toggle = not self._toggle
        jitter_x = target_joy_x + (1 if self._toggle else -1)

        now = self.time_provider()
        if not self.is_holding or (now - self.last_down_time > 1.8):
            self.transport.send_motionevent("DOWN", self.center_x, self.center_y)
            self.transport.send_motionevent("MOVE", target_joy_x, target_joy_y)
            self.is_holding = True
            self.last_down_time = now
        else:
            self.transport.send_motionevent("MOVE", jitter_x, target_joy_y)

        self.current_pos = (target_joy_x, target_joy_y)
        return target_joy_x, target_joy_y

    def kite_away(
        self,
        hero_pos: Tuple[float, float],
        danger_pos: Tuple[float, float],
        duration_ms: int = 400,
    ) -> Tuple[int, int]:
        """
        Directs the virtual joystick away from danger_pos (opposite direction vector).
        """
        hx, hy = hero_pos
        ex, ey = danger_pos

        dx = hx - ex
        dy = hy - ey
        angle = math.atan2(dy, dx)

        target_joy_x = int(round(self.center_x + self.radius * math.cos(angle)))
        target_joy_y = int(round(self.center_y + self.radius * math.sin(angle)))

        self._toggle = not self._toggle
        jitter_x = target_joy_x + (1 if self._toggle else -1)

        now = self.time_provider()
        if not self.is_holding or (now - self.last_down_time > 1.8):
            self.transport.send_motionevent("DOWN", self.center_x, self.center_y)
            self.transport.send_motionevent("MOVE", target_joy_x, target_joy_y)
            self.is_holding = True
            self.last_down_time = now
        else:
            self.transport.send_motionevent("MOVE", jitter_x, target_joy_y)

        self.current_pos = (target_joy_x, target_joy_y)
        return target_joy_x, target_joy_y

    def release(self) -> bool:
        """Lifts finger from the virtual joystick."""
        if self.is_holding:
            cx, cy = self.current_pos
            success = self.transport.send_motionevent("UP", cx, cy)
            self.is_holding = False
            self.current_pos = (self.center_x, self.center_y)
            return bool(success)
        return True

    def close(self) -> None:
        """Safely releases joystick."""
        self.release()
