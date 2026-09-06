"""
brain.state — Immutable minimal tactical state extracted from observed WorldState.

Extracts only the distilled numerical features and flags required by tactical decision models:
- Distances to nearest enemy/minion
- Health ratios
- Skill/combo availability
- Entity counts
- Proximity to defensive turrets

Pure feature extraction layer: strictly immutable, no decisions, no hardware dependencies.
"""

from dataclasses import dataclass
from typing import Optional, List
import math

from world.models import Vector2, WorldState, EnemyState, MinionState, TurretState
from config.config import TURRET_DANGER_DIST_PX, SCREEN_WIDTH, SCREEN_HEIGHT


@dataclass(frozen=True)
class TacticalState:
    """
    Distilled, immutable tactical representation of the instantaneous battlefield.
    Supplies all inputs required by Tactical Brains without exposing raw vision abstractions.
    """
    timestamp: float
    player_position: Vector2
    player_hp_ratio: float
    is_player_dead: bool

    s1_ready: bool
    s2_ready: bool
    ultimate_ready: bool
    can_combo: bool

    enemy_visible: bool
    enemy_count: int
    nearest_enemy_distance: Optional[float]
    nearest_enemy_position: Optional[Vector2]

    minion_count: int
    nearest_minion_distance: Optional[float]
    nearest_minion_position: Optional[Vector2]

    enemy_near_turret: bool
    is_near_base: bool
    nearest_enemy_id: Optional[int] = None
    nearest_minion_id: Optional[int] = None
    is_player_visible: bool = True


class TacticalStateBuilder:
    """
    Constructs an immutable TacticalState from an observed WorldState.
    Performs pure feature extraction without making tactical decisions.
    """

    def __init__(
        self,
        turret_danger_dist: float = TURRET_DANGER_DIST_PX,
        default_player_pos: Optional[Vector2] = None,
    ):
        self.turret_danger_dist = float(turret_danger_dist)
        self.default_player_pos = (
            default_player_pos
            if default_player_pos is not None
            else Vector2(SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0)
        )

    def build(self, world_state: WorldState) -> TacticalState:
        """
        Extracts distilled numerical features from WorldState.

        Args:
            world_state: Unified battlefield state from World layer.

        Returns:
            TacticalState: Immutable feature vector for Tactical Brain.
        """
        if not isinstance(world_state, WorldState):
            raise TypeError(f"TacticalStateBuilder expects WorldState, got: {type(world_state)}")

        # 1. Player state extraction
        if world_state.player:
            player_pos = world_state.player.position
            is_visible = bool(world_state.player.is_visible)
            hp_ratio = float(world_state.player.hp.value) if world_state.player.hp else 1.0
            # LOST != DEAD: Temporary loss of vision/missing frames must never automatically mean player death.
            # Confirmed player death requires explicit death flag or zero HP.
            is_dead = getattr(world_state.player, "is_dead", False) or (
                world_state.player.hp is not None
                and world_state.player.hp.value <= 0.0
            )
        else:
            player_pos = self.default_player_pos
            is_visible = False
            is_dead = False
            hp_ratio = 1.0

        # 2. Skill state extraction
        if world_state.skills:
            s1_ok = bool(world_state.skills.s1_ready)
            s2_ok = bool(world_state.skills.s2_ready)
            ult_ok = bool(world_state.skills.ultimate_ready)
        else:
            s1_ok = True
            s2_ok = True
            ult_ok = True

        can_combo = s2_ok and ult_ok

        # 3. Enemy feature extraction
        enemy_count = len(world_state.enemies)
        enemy_visible = enemy_count > 0

        nearest_enemy_dist: Optional[float] = None
        nearest_enemy_pos: Optional[Vector2] = None
        nearest_enemy_id: Optional[int] = None

        if world_state.enemies:
            closest_enemy = min(
                world_state.enemies,
                key=lambda e: player_pos.distance_to(e.position),
            )
            nearest_enemy_dist = player_pos.distance_to(closest_enemy.position)
            nearest_enemy_pos = closest_enemy.position
            nearest_enemy_id = getattr(closest_enemy, "track_id", None)

        # 4. Enemy minion feature extraction
        enemy_minions = [m for m in world_state.minions if m.is_enemy]
        minion_count = len(enemy_minions)

        nearest_minion_dist: Optional[float] = None
        nearest_minion_pos: Optional[Vector2] = None
        nearest_minion_id: Optional[int] = None

        if enemy_minions:
            closest_minion = min(
                enemy_minions,
                key=lambda m: player_pos.distance_to(m.position),
            )
            nearest_minion_dist = player_pos.distance_to(closest_minion.position)
            nearest_minion_pos = closest_minion.position
            nearest_minion_id = getattr(closest_minion, "track_id", None)

        # 5. Turret proximity
        turret_near = any(
            t.is_enemy and player_pos.distance_to(t.position) < self.turret_danger_dist
            for t in world_state.turrets
        )

        return TacticalState(
            timestamp=world_state.timestamp,
            player_position=player_pos,
            player_hp_ratio=hp_ratio,
            is_player_dead=is_dead,
            s1_ready=s1_ok,
            s2_ready=s2_ok,
            ultimate_ready=ult_ok,
            can_combo=can_combo,
            enemy_visible=enemy_visible,
            enemy_count=enemy_count,
            nearest_enemy_distance=nearest_enemy_dist,
            nearest_enemy_position=nearest_enemy_pos,
            minion_count=minion_count,
            nearest_minion_distance=nearest_minion_dist,
            nearest_minion_position=nearest_minion_pos,
            enemy_near_turret=turret_near,
            is_near_base=bool(world_state.is_near_base),
            nearest_enemy_id=nearest_enemy_id,
            nearest_minion_id=nearest_minion_id,
            is_player_visible=is_visible,
        )
