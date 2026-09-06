"""
control.attack — Authoritative V2 Attack Controller implementing AttackPort.

Dispatches targeted entity attacks through centralized ADBTransport.
Applies biomechanical humanization and enforces cooldown pacing without tactical target choice.
"""

import time
from typing import Optional, Callable

from control.interfaces import AttackPort, JoystickPort
from control.adb import ADBTransport
from control.humanizer import InputHumanizer
from config.config import BASIC_ATTACK_COOLDOWN_SEC


class AttackControl(AttackPort):
    """
    V2 Attack Controller implementing AttackPort.
    Dispatches basic attack swipes/taps with anti-detection humanization and cooldown pacing.
    """

    # Calibrated coordinates for basic attack button in 1544x720 landscape
    DEFAULT_ATTACK_X: int = 1350
    DEFAULT_ATTACK_Y: int = 560

    def __init__(
        self,
        transport: ADBTransport,
        humanizer: Optional[InputHumanizer] = None,
        joystick: Optional[JoystickPort] = None,
        cooldown_sec: float = BASIC_ATTACK_COOLDOWN_SEC,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.transport = transport
        self.humanizer = humanizer or InputHumanizer()
        self.joystick = joystick
        self.cooldown_sec = float(cooldown_sec)
        self.time_provider = time_provider or time.time
        self.last_attack_time: float = 0.0

    def attack(self, target_id: Optional[int] = None) -> bool:
        """
        Dispatches basic attack with humanized coordinates and hold duration.
        Target_id is preserved for logging/tracking without tactical inference.

        Returns:
            True if attack was dispatched, False if on cooldown or transport failed.
        """
        now = self.time_provider()
        if (now - self.last_attack_time) < self.cooldown_sec:
            return False

        jx, jy = self.humanizer.jitter_point(
            self.DEFAULT_ATTACK_X,
            self.DEFAULT_ATTACK_Y,
            std_dev=2.0,
            max_offset=4,
        )
        hold_ms = self.humanizer.hold_duration_ms(
            mean_ms=45.0,
            std_ms=6.0,
            min_ms=30.0,
            max_ms=65.0,
        )

        success = self.transport.send_swipe(jx, jy, jx, jy, duration_ms=hold_ms)
        self.last_attack_time = now

        if self.joystick and hasattr(self.joystick, "notify_screen_touched"):
            self.joystick.notify_screen_touched()

        return bool(success)
