"""
actions.executor — Bridge layer between abstract Action models and concrete Control mechanisms.

Translates intent (Action) into delegated invocations of injected control interfaces
(joystick, skills macro, attack adapter).
Fully isolated from hardware details, screen coordinates, subprocesses, and ADB.
"""

from dataclasses import dataclass
from typing import Optional, Any, Callable, Tuple

from actions.models import Action, ActionType
from world.models import Vector2


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
    Executes abstract Actions through injected control components.
    Zero hardware dependencies; zero hardcoded button coordinates.
    """

    def __init__(
        self,
        joystick: Optional[Any] = None,
        skills: Optional[Any] = None,
        attack_controller: Optional[Any] = None,
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

        if action.type in (ActionType.RETREAT, ActionType.FARM):
            return self.joystick is not None or self.skills is not None

        return False

    def execute(self, action: Action) -> ExecutionResult:
        """
        Dispatches action to the appropriate control component.

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

    def _execute_move(self, action: Action) -> ExecutionResult:
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
        if not self.joystick:
            return ExecutionResult(success=False, action=action, error="Joystick control not configured")

        hero_pos = self.hero_pos_provider()
        if action.direction is not None:
            danger_pos = (action.direction.x, action.direction.y)
        else:
            danger_pos = (hero_pos[0] + 100, hero_pos[1])

        duration_ms = self._get_duration_ms(action)
        if hasattr(self.joystick, "kite_away"):
            self.joystick.kite_away(hero_pos, danger_pos, duration_ms=duration_ms)
        elif hasattr(self.joystick, "move_towards"):
            self.joystick.move_towards(hero_pos, danger_pos, duration_ms=duration_ms)

        return ExecutionResult(success=True, action=action)

    def _execute_retreat(self, action: Action) -> ExecutionResult:
        executed = False
        # Optional panic retreat skill if available
        if self.skills and hasattr(self.skills, "panic_retreat_s2"):
            self.skills.panic_retreat_s2()
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
        if self.attack_controller and hasattr(self.attack_controller, "attack"):
            self.attack_controller.attack(target_id=action.target_id)
            return ExecutionResult(success=True, action=action)

        if self.skills:
            if hasattr(self.skills, "fire_basic_attack_fast"):
                self.skills.fire_basic_attack_fast()
                return ExecutionResult(success=True, action=action)
            elif hasattr(self.skills, "trigger_basic_attack"):
                self.skills.trigger_basic_attack()
                return ExecutionResult(success=True, action=action)

        return ExecutionResult(
            success=False,
            action=action,
            error="No attack adapter or skills controller available"
        )

    def _execute_cast_s1(self, action: Action) -> ExecutionResult:
        if not self.skills:
            return ExecutionResult(success=False, action=action, error="Skills controller not configured")

        if hasattr(self.skills, "fire_skill_1_fast"):
            self.skills.fire_skill_1_fast()
        elif hasattr(self.skills, "cast_s1"):
            self.skills.cast_s1()
        else:
            return ExecutionResult(success=False, action=action, error="Skills controller lacks S1 method")

        return ExecutionResult(success=True, action=action)

    def _execute_cast_s2(self, action: Action) -> ExecutionResult:
        if not self.skills:
            return ExecutionResult(success=False, action=action, error="Skills controller not configured")

        if hasattr(self.skills, "fire_skill_2_fast"):
            self.skills.fire_skill_2_fast()
        elif hasattr(self.skills, "cast_s2"):
            self.skills.cast_s2()
        else:
            return ExecutionResult(success=False, action=action, error="Skills controller lacks S2 method")

        return ExecutionResult(success=True, action=action)

    def _execute_cast_ult(self, action: Action) -> ExecutionResult:
        if not self.skills:
            return ExecutionResult(success=False, action=action, error="Skills controller not configured")

        if hasattr(self.skills, "trigger_engage_ultimate"):
            self.skills.trigger_engage_ultimate()
        elif hasattr(self.skills, "cast_ult"):
            self.skills.cast_ult()
        else:
            return ExecutionResult(success=False, action=action, error="Skills controller lacks ULT method")

        return ExecutionResult(success=True, action=action)

    def _execute_farm(self, action: Action) -> ExecutionResult:
        if self.attack_controller and hasattr(self.attack_controller, "attack"):
            self.attack_controller.attack(target_id=action.target_id)
        elif self.skills and hasattr(self.skills, "fire_basic_attack_fast"):
            self.skills.fire_basic_attack_fast()
        elif not self.joystick:
            return ExecutionResult(success=False, action=action, error="No control available for FARM")

        if self.joystick and action.direction is not None:
            hero_pos = self.hero_pos_provider()
            target_pos = (action.direction.x, action.direction.y)
            duration_ms = self._get_duration_ms(action)
            if hasattr(self.joystick, "move_towards"):
                self.joystick.move_towards(hero_pos, target_pos, duration_ms=duration_ms)

        return ExecutionResult(success=True, action=action)

    def _execute_scout(self, action: Action) -> ExecutionResult:
        return self._execute_move(action)

    def _execute_idle(self, action: Action) -> ExecutionResult:
        return ExecutionResult(success=True, action=action)
