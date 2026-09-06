"""
control.skills — Authoritative V2 Skill & Combo Controller implementing SkillsPort.

Dispatches skill casts and true Claude engagement combos through centralized ADBTransport.
Applies biomechanical humanization and microsecond timing.
Preserves true combo semantics:
S2 (Place Dexter) -> S2 (Teleport In) -> S1 (Art of Thievery) -> Ultimate (Blazing Duet)
Strictly 4 steps (no fifth S2-back step).
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass

from control.interfaces import SkillsPort, JoystickPort
from control.adb import ADBTransport
from control.humanizer import InputHumanizer
from config.config import (
    SPATIAL_MAX_OFFSET,
    HOLD_DURATION_MEAN_MS,
    HOLD_DURATION_MIN_MS,
    HOLD_DURATION_MAX_MS,
)


@dataclass(frozen=True)
class SkillActionDef:
    name: str
    coords: Tuple[int, int]
    delay_after_ms: float
    hold_duration_ms: float


class SkillsControl(SkillsPort):
    """
    V2 Skills & Macro Controller implementing SkillsPort.
    Executes individual skill casts and chained macro combos using ADBTransport.
    """

    # Calibrated HUD coordinates in 1544x720 landscape
    COORD_SKILL_1: Tuple[int, int] = (1088, 540)
    COORD_SKILL_2: Tuple[int, int] = (1228, 392)
    COORD_ULTIMATE: Tuple[int, int] = (1051, 375)
    COORD_BASIC_ATTACK: Tuple[int, int] = (1350, 560)
    COORD_REGEN: Tuple[int, int] = (815, 640)

    def __init__(
        self,
        transport: ADBTransport,
        humanizer: Optional[InputHumanizer] = None,
        joystick: Optional[JoystickPort] = None,
    ):
        self.transport = transport
        self.humanizer = humanizer or InputHumanizer()
        self.joystick = joystick

    def _notify_joystick(self) -> None:
        if self.joystick and hasattr(self.joystick, "notify_screen_touched"):
            try:
                self.joystick.notify_screen_touched()
            except Exception:
                pass

    def fire_basic_attack_fast(self) -> bool:
        """Dispatches high-speed basic attack swipe."""
        jx, jy = self.humanizer.jitter_point(
            self.COORD_BASIC_ATTACK[0],
            self.COORD_BASIC_ATTACK[1],
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
        self._notify_joystick()
        return bool(success)

    def fire_skill_1_fast(self) -> bool:
        """Dispatches Skill 1 (Art of Thievery)."""
        jx, jy = self.humanizer.jitter_point(
            self.COORD_SKILL_1[0],
            self.COORD_SKILL_1[1],
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
        self._notify_joystick()
        return bool(success)

    def fire_skill_2_fast(self) -> bool:
        """Dispatches Skill 2 (Battle Mirror Image)."""
        jx, jy = self.humanizer.jitter_point(
            self.COORD_SKILL_2[0],
            self.COORD_SKILL_2[1],
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
        self._notify_joystick()
        return bool(success)

    def panic_retreat_s2(self) -> bool:
        """
        Dispatches panic escape sequence on Skill 2: double-tap (drop shadow + immediate swap).
        """
        actions = [
            SkillActionDef(
                name="Panic S2 (Place Shadow)",
                coords=self.COORD_SKILL_2,
                delay_after_ms=120.0,
                hold_duration_ms=45.0,
            ),
            SkillActionDef(
                name="Panic S2 (Swap to Safety)",
                coords=self.COORD_SKILL_2,
                delay_after_ms=0.0,
                hold_duration_ms=45.0,
            ),
        ]
        return self._execute_sequence(actions)

    def trigger_engage_ultimate(self, ult_channel_ms: float = 1000.0) -> bool:
        """
        True Claude Combo (Dive & Burst):
        1. Skill 2 (Place Dexter): drop shadow decoy
        2. Skill 2 (Teleport In): dive into target
        3. Skill 1 (Art of Thievery): steal attack speed to ramp up ult
        4. Ultimate (Blazing Duet): channel burst rotation

        Strictly 4 steps: preserves verified absence of 5th S2-back step.
        """
        actions = [
            SkillActionDef(
                name="Skill 2 (Place Dexter)",
                coords=self.COORD_SKILL_2,
                delay_after_ms=120.0,
                hold_duration_ms=45.0,
            ),
            SkillActionDef(
                name="Skill 2 (Teleport In)",
                coords=self.COORD_SKILL_2,
                delay_after_ms=50.0,
                hold_duration_ms=40.0,
            ),
            SkillActionDef(
                name="Skill 1 (Art of Thievery)",
                coords=self.COORD_SKILL_1,
                delay_after_ms=40.0,
                hold_duration_ms=40.0,
            ),
            SkillActionDef(
                name="Ultimate (Blazing Duet)",
                coords=self.COORD_ULTIMATE,
                delay_after_ms=float(ult_channel_ms),
                hold_duration_ms=55.0,
            ),
        ]
        return self._execute_sequence(actions)

    def _execute_sequence(self, actions: List[SkillActionDef]) -> bool:
        """
        Chains a sequence of actions with low-level usleep delays into a single shell command.
        Dispatches through persistent ADBTransport.
        """
        commands = []
        for i, action in enumerate(actions):
            jx, jy = self.humanizer.jitter_point(
                action.coords[0],
                action.coords[1],
                std_dev=3.0,
                max_offset=SPATIAL_MAX_OFFSET,
            )
            hold_ms = int(round(action.hold_duration_ms))
            commands.append(f"input swipe {jx} {jy} {jx} {jy} {hold_ms}")

            if i < len(actions) - 1:
                delay_ms = action.delay_after_ms
                jittered_delay = self.humanizer.reaction_delay_ms(
                    base_ms=delay_ms,
                    variance_ms=max(1.0, delay_ms * 0.08),
                    min_floor_ms=5.0,
                )
                delay_us = int(jittered_delay * 1000)
                commands.append(f"usleep {delay_us}")

        chained_shell_cmd = " && ".join(commands)
        success = self.transport.send(chained_shell_cmd)
        self._notify_joystick()
        return bool(success)
