"""
actions.models — Pure abstract action data contract between Tactical Brain and Control layer.

Actions represent intent ("what to do") without any knowledge of Android buttons,
touch screen coordinates, ADB commands, or device drivers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math

from world.models import Vector2


class ActionType(str, Enum):
    """
    Abstract battlefield actions corresponding to V1/V2 operational modes.
    """
    MOVE = "move"
    ATTACK = "attack"
    CAST_S1 = "cast_s1"
    CAST_S2 = "cast_s2"
    CAST_ULT = "cast_ult"
    RETREAT = "retreat"
    FARM = "farm"
    KITE = "kite"
    SCOUT = "scout"
    IDLE = "idle"


@dataclass(frozen=True)
class Action:
    """
    Immutable specification of an intended bot command.

    Attributes:
        type: Action category (MOVE, ATTACK, CAST_S1, etc.).
        direction: Optional 2D target vector or movement direction (in reference space).
        target_id: Optional track ID of entity being engaged.
        duration: Optional action hold duration in seconds (e.g., movement step).
    """
    type: ActionType
    direction: Optional[Vector2] = None
    target_id: Optional[int] = None
    duration: Optional[float] = None

    def __post_init__(self):
        """Validates action structural integrity."""
        if not isinstance(self.type, ActionType):
            raise TypeError(f"Action type must be an ActionType enum, got: {type(self.type)}")

        if self.duration is not None:
            if math.isnan(self.duration) or self.duration < 0.0:
                raise ValueError(f"Action duration must be non-negative, got: {self.duration}")

        if self.direction is not None:
            if math.isnan(self.direction.x) or math.isnan(self.direction.y):
                raise ValueError(f"Action direction contains NaN coordinates: {self.direction}")
            if math.isinf(self.direction.x) or math.isinf(self.direction.y):
                raise ValueError(f"Action direction contains infinite coordinates: {self.direction}")

        if self.target_id is not None and not isinstance(self.target_id, int):
            raise TypeError(f"target_id must be int or None, got: {type(self.target_id)}")
