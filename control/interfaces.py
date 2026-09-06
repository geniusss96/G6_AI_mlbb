"""
control.interfaces — Abstract control ports and protocol definitions.

Defines the required control interface contracts expected by ActionExecutor.
These protocols maintain loose coupling between Action execution and physical/mock hardware.
"""

from typing import Protocol, runtime_checkable, Optional, Tuple, Any


@runtime_checkable
class JoystickPort(Protocol):
    """Abstract port for directional movement and joystick control."""

    def move_towards(
        self,
        hero_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        duration_ms: int = 400,
    ) -> Any: ...

    def kite_away(
        self,
        hero_pos: Tuple[float, float],
        danger_pos: Tuple[float, float],
        duration_ms: int = 400,
    ) -> Any: ...

    def notify_screen_touched(self) -> None: ...


@runtime_checkable
class SkillsPort(Protocol):
    """Abstract port for skill and attack macro execution."""

    def fire_basic_attack_fast(self) -> Any: ...

    def fire_skill_1_fast(self) -> Any: ...

    def fire_skill_2_fast(self) -> Any: ...

    def trigger_engage_ultimate(self) -> Any: ...

    def panic_retreat_s2(self) -> Any: ...


@runtime_checkable
class AttackPort(Protocol):
    """Abstract port for targeted entity attacks."""

    def attack(self, target_id: Optional[int] = None) -> Any: ...
