"""
actions.executor — Thin execution dispatcher from V2 Action intent to low-level control primitives.

Translates intent (Action) into delegated invocations of injected control interfaces
(joystick, skills macro, attack adapter).
Zero tactical decisions: tactical decisions belong to TacticalBrain / ActionSelector.
Fully isolated from hardware details, screen coordinates, subprocesses, and ADB.
"""

from dataclasses import dataclass
from typing import Optional, Any, Callable, Tuple

from actions.models import Action, ActionType
from control.interfaces import JoystickPort, SkillsPort, AttackPort


@dataclass(frozen=True)
class ExecutionResult:
    """
    Diagnostic result of executing an abstract Action through control interfaces.
    """
    success: bool
    action: Action
    error: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


class ActionExecutor:
    """
    Thin dispatcher that maps abstract Action intent to low-level control operations.
    Zero tactical decision logic; zero hardware dependencies; zero hardcoded button coordinates.
    """

    def __init__(
        self,
        joystick: Optional[JoystickPort] = None,
        skills: Optional[SkillsPort] = None,
        attack_controller: Optional[AttackPort] = None,
        humanizer: Optional[Any] = None,
        hero_pos_provider: Optional[Callable[[], Tuple[float, float]]] = None,
    ):
        self.joystick = joystick
        self.skills = skills
        self.attack_controller = attack_controller
        self.humanizer = humanizer
        self.hero_pos_provider = hero_pos_provider or (lambda: (772.0, 360.0))

    def can_execute(self, action: Action) -> bool:
        """
        Validates whether required control dependencies are available to execute action.
        Contains no tactical logic.
        """
        if not isinstance(action, Action):
            return False

        if action.type == ActionType.IDLE:
            return True

        if action.type in (ActionType.MOVE, ActionType.SCOUT, ActionType.KITE):
            return self.joystick is not None

        if action.type in (ActionType.CAST_S1, ActionType.CAST_S2, ActionType.CAST_ULT):
            return self.skills is not None

        if action.type == ActionType.ATTACK:
            return self.attack_controller is not None or self.skills is not None

        if action.type == ActionType.RETREAT:
            return self.joystick is not None or self.skills is not None

        if action.type == ActionType.FARM:
            return (
                self.attack_controller is not None
                or self.skills is not None
                or self.joystick is not None
            )

        return False

    def execute(self, action: Action) -> ExecutionResult:
        """
        Dispatches action to the appropriate control component.
        Does not mutate action intent or make tactical adjustments.

        Returns:
            ExecutionResult indicating success or failure.
        """
        if not isinstance(action, Action):
            return ExecutionResult(
                success=False,
                action=action,  # type: ignore
                error=f"Invalid action object: {type(action)}"
            )

        try:
            handler = self._get_handler(action.type)
            if handler is None:
                return ExecutionResult(
                    success=False,
                    action=action,
                    error=f"No handler registered for action type {action.type}"
                )
            return handler(action)
        except Exception as e:
            return ExecutionResult(
                success=False,
                action=action,
                error=str(e)
            )

    def _get_handler(self, action_type: ActionType):
        handlers = {
            ActionType.MOVE: self._execute_move,
            ActionType.KITE: self._execute_kite,
            ActionType.RETREAT: self._execute_retreat,
            ActionType.ATTACK: self._execute_attack,
            ActionType.CAST_S1: self._execute_cast_s1,
            ActionType.CAST_S2: self._execute_cast_s2,
            ActionType.CAST_ULT: self._execute_cast_ult,
            ActionType.FARM: self._execute_farm,
            ActionType.SCOUT: self._execute_scout,
            ActionType.IDLE: self._execute_idle,
        }
        return handlers.get(action_type)

    def _get_duration_ms(self, action: Action, default_ms: int = 400) -> int:
        if action.duration is not None:
            return max(10, int(action.duration * 1000))
        return default_ms

    def _notify_touch(self) -> None:
        """
        Notifies joystick touch-watchdog that a screen interaction occurred.
        Resets touch-holding state in joystick to prevent character paralysis.
        """
        if self.joystick and hasattr(self.joystick, "notify_screen_touched"):
            try:
                self.joystick.notify_screen_touched()
            except Exception:
                pass

    def _execute_move(self, action: Action) -> ExecutionResult:
        """Dispatches movement control."""
        if not self.joystick:
            return ExecutionResult(success=False, action=action, error="Joystick control not configured")

        hero_pos = self.hero_pos_provider()
        if action.direction is not None:
            target_pos = (action.direction.x, action.direction.y)
        else:
            target_pos = hero_pos

        duration_ms = self._get_duration_ms(action)
        if hasattr(self.joystick, "move_towards"):
            self.joystick.move_towards(hero_pos, target_pos, duration_ms=duration_ms)
        elif hasattr(self.joystick, "move"):
            self.joystick.move(target_pos, duration_ms=duration_ms)

        return ExecutionResult(success=True, action=action)

    def _execute_kite(self, action: Action) -> ExecutionResult:
        """
        Dispatches kite movement away from danger vector encoded in action.
        Does not invent fake tactical danger coordinates.
        """
        if not self.joystick:
            return ExecutionResult(success=False, action=action, error="Joystick control not configured")

        if action.direction is None:
            return ExecutionResult(
                success=False,
                action=action,
                error="KITE requires direction vector to kite away from"
            )

        hero_pos = self.hero_pos_provider()
        danger_pos = (action.direction.x, action.direction.y)

        duration_ms = self._get_duration_ms(action)
        if hasattr(self.joystick, "kite_away"):
            self.joystick.kite_away(hero_pos, danger_pos, duration_ms=duration_ms)
        elif hasattr(self.joystick, "move_towards"):
            self.joystick.move_towards(hero_pos, danger_pos, duration_ms=duration_ms)

        return ExecutionResult(success=True, action=action)

    def _execute_retreat(self, action: Action) -> ExecutionResult:
        """
        Dispatches retreat mechanics: panic retreat escape skill if available,
        and directional retreat movement via joystick if direction is supplied.
        Does not make independent tactical decisions to retreat or attack.
        """
        executed = False
        # Optional panic retreat skill if available
        if self.skills and hasattr(self.skills, "panic_retreat_s2"):
            self.skills.panic_retreat_s2()
            self._notify_touch()
            executed = True

        if self.joystick and action.direction is not None:
            hero_pos = self.hero_pos_provider()
            target_pos = (action.direction.x, action.direction.y)
            duration_ms = self._get_duration_ms(action)
            if hasattr(self.joystick, "move_towards"):
                self.joystick.move_towards(hero_pos, target_pos, duration_ms=duration_ms)
            executed = True

        if not executed:
            return ExecutionResult(
                success=False,
                action=action,
                error="Neither skills nor joystick available for retreat"
            )

        return ExecutionResult(success=True, action=action)

    def _execute_attack(self, action: Action) -> ExecutionResult:
        """
        Dispatches attack control to attack adapter or skills macro.
        Notifies joystick watchdog of screen touch.
        """
        if self.attack_controller and hasattr(self.attack_controller, "attack"):
            self.attack_controller.attack(target_id=action.target_id)
            self._notify_touch()
            return ExecutionResult(success=True, action=action)

        if self.skills:
            if hasattr(self.skills, "fire_basic_attack_fast"):
                self.skills.fire_basic_attack_fast()
                self._notify_touch()
                return ExecutionResult(success=True, action=action)
            elif hasattr(self.skills, "trigger_basic_attack"):
                self.skills.trigger_basic_attack()
                self._notify_touch()
                return ExecutionResult(success=True, action=action)

        return ExecutionResult(
            success=False,
            action=action,
            error="No attack adapter or skills controller available"
        )

    def _execute_cast_s1(self, action: Action) -> ExecutionResult:
        """Dispatches Skill 1 cast."""
        if not self.skills:
            return ExecutionResult(success=False, action=action, error="Skills controller not configured")

        if hasattr(self.skills, "fire_skill_1_fast"):
            self.skills.fire_skill_1_fast()
        elif hasattr(self.skills, "cast_s1"):
            self.skills.cast_s1()
        else:
            return ExecutionResult(success=False, action=action, error="Skills controller lacks S1 method")

        self._notify_touch()
        return ExecutionResult(success=True, action=action)

    def _execute_cast_s2(self, action: Action) -> ExecutionResult:
        """Dispatches Skill 2 cast."""
        if not self.skills:
            return ExecutionResult(success=False, action=action, error="Skills controller not configured")

        if hasattr(self.skills, "fire_skill_2_fast"):
            self.skills.fire_skill_2_fast()
        elif hasattr(self.skills, "cast_s2"):
            self.skills.cast_s2()
        else:
            return ExecutionResult(success=False, action=action, error="Skills controller lacks S2 method")

        self._notify_touch()
        return ExecutionResult(success=True, action=action)

    def _execute_cast_ult(self, action: Action) -> ExecutionResult:
        """
        Dispatches Ultimate combo/cast.
        Preserves optimized S2 -> S2 -> S1 -> Ult combo timing via trigger_engage_ultimate.
        """
        if not self.skills:
            return ExecutionResult(success=False, action=action, error="Skills controller not configured")

        if hasattr(self.skills, "trigger_engage_ultimate"):
            self.skills.trigger_engage_ultimate()
        elif hasattr(self.skills, "cast_ult"):
            self.skills.cast_ult()
        else:
            return ExecutionResult(success=False, action=action, error="Skills controller lacks ULT method")

        self._notify_touch()
        return ExecutionResult(success=True, action=action)

    def _execute_farm(self, action: Action) -> ExecutionResult:
        """
        Dispatches farming primitives: attack on entity and/or positioning towards minion.
        Does NOT decide independently whether to retreat, kite, or engage.
        """
        executed = False
        if self.attack_controller and hasattr(self.attack_controller, "attack"):
            self.attack_controller.attack(target_id=action.target_id)
            self._notify_touch()
            executed = True
        elif self.skills and hasattr(self.skills, "fire_basic_attack_fast"):
            self.skills.fire_basic_attack_fast()
            self._notify_touch()
            executed = True

        if self.joystick and action.direction is not None:
            hero_pos = self.hero_pos_provider()
            target_pos = (action.direction.x, action.direction.y)
            duration_ms = self._get_duration_ms(action)
            if hasattr(self.joystick, "move_towards"):
                self.joystick.move_towards(hero_pos, target_pos, duration_ms=duration_ms)
            executed = True

        if not executed:
            return ExecutionResult(
                success=False,
                action=action,
                error="No control available for FARM execution"
            )

        return ExecutionResult(success=True, action=action)

    def _execute_scout(self, action: Action) -> ExecutionResult:
        """
        Dispatches scouting movement using movement control.
        Contains zero routing heuristics or tactical decisions.
        """
        return self._execute_move(action)

    def _execute_idle(self, action: Action) -> ExecutionResult:
        """Explicit no-op for idle action."""
        return ExecutionResult(success=True, action=action)
