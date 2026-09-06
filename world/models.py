"""
world.models — Clean data models representing the observed battlefield state.

Strictly immutable dataclasses for entities, player state, and the unified WorldState.
Does NOT contain business logic, tracking hysteresis, corpse timers, tactical decisions,
or device communication.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math

from vision.skill_state import SkillState
from vision.hp_detector import HPObservation


@dataclass(frozen=True)
class Vector2:
    """
    2D coordinate vector in canonical reference system (1544x720).
    """
    x: float
    y: float

    def distance_to(self, other: "Vector2") -> float:
        """Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> Tuple[float, float]:
        """Convert to (x, y) tuple."""
        return (self.x, self.y)


@dataclass(frozen=True)
class PlayerState:
    """
    Observed state of our hero (Claude) in current frame.
    """
    position: Vector2
    hp: HPObservation
    is_visible: bool = True


@dataclass(frozen=True)
class EnemyState:
    """
    Observed enemy hero on the battlefield in current frame.
    """
    position: Vector2
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    source: str = "yolo"  # "yolo" or "optical_hp"


@dataclass(frozen=True)
class MinionState:
    """
    Observed lane minion, jungle creep, or neutral buff in current frame.
    """
    position: Vector2
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    is_enemy: bool = True
    kind: str = "minion"  # "minion", "buff", "crab", etc.


@dataclass(frozen=True)
class TurretState:
    """
    Observed defensive turret in current frame.
    """
    position: Vector2
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    is_enemy: bool = True


@dataclass
class WorldState:
    """
    Unified contract representing instantaneous observed state of the battlefield.
    Passed downstream from Vision to Tracker and Tactical Brain.
    """
    timestamp: float
    player: Optional[PlayerState] = None
    enemies: List[EnemyState] = field(default_factory=list)
    minions: List[MinionState] = field(default_factory=list)
    turrets: List[TurretState] = field(default_factory=list)
    skills: Optional[SkillState] = None
    is_near_base: bool = False
