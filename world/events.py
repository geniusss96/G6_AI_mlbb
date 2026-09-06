"""
world.events — Watchdogs and spatial suppression zone management.

Isolates:
  1. Anti-Corpse Watchdog (flags stationary targets >2.5s as dead/textures).
  2. Ally-Creep Watchdog (flags invulnerable creeps under continuous attack >4.5s).
  3. Spatial suppression zones (corpse / ally blacklist with 20.0s expiration).

Strictly deterministic: uses explicit timestamps passed in update().
Does NOT decide tactical actions, does NOT grant rewards, and does NOT
declare hero death simply based on track loss.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from world.models import Vector2
from world.tracker import Track


@dataclass(frozen=True)
class SuppressionZone:
    """
    Temporary spatial blacklist zone (e.g. corpse or invincible ally creep).
    """
    position: Vector2
    radius_px: float
    expiry: float
    reason: str  # "CORPSE", "INVULNERABLE_CREEP", etc.

    def contains(self, point: Vector2) -> bool:
        return self.position.distance_to(point) <= self.radius_px


@dataclass
class TargetStationaryState:
    """
    State tracking how long an attacked entity has remained motionless.
    """
    target_id: Optional[int]
    last_position: Vector2
    stationary_since: float


@dataclass
class AttackEngagementState:
    """
    State tracking duration of continuous attack against an entity.
    """
    target_id: Optional[int]
    attack_started_at: float
    last_attacked_at: float


class WatchdogManager:
    """
    Manages anti-corpse and ally-creep watchdogs with spatial suppression zones.
    Preserves exact V1 thresholds and temporal behavior.
    """

    CORPSE_STATIONARY_TIMEOUT_SEC = 2.5
    CORPSE_STATIONARY_RADIUS_PX = 20.0
    CORPSE_ZONE_RADIUS_PX = 60.0
    CORPSE_SUPPRESSION_DURATION_SEC = 20.0

    ALLY_CREEP_ATTACK_TIMEOUT_SEC = 4.5
    ALLY_CREEP_ZONE_RADIUS_PX = 60.0
    ALLY_CREEP_SUPPRESSION_DURATION_SEC = 20.0

    def __init__(self):
        self._suppression_zones: List[SuppressionZone] = []
        self._enemy_stationary: Optional[TargetStationaryState] = None
        self._creep_attack: Optional[AttackEngagementState] = None

    def clean_expired(self, timestamp: float) -> None:
        """Removes suppression zones whose expiry timestamp has passed."""
        self._suppression_zones = [z for z in self._suppression_zones if timestamp < z.expiry]

    def is_suppressed(self, point: Vector2) -> bool:
        """
        Checks if a spatial coordinate falls within any active suppression zone.
        """
        return any(zone.contains(point) for zone in self._suppression_zones)

    def update_enemy_stationary(
        self,
        target_pos: Optional[Vector2],
        timestamp: float,
        target_id: Optional[int] = None
    ) -> bool:
        """
        ANTI-CORPSE WATCHDOG:
        Monitors whether the enemy target remains in place (<20px movement) for >2.5s.

        Args:
            target_pos: Current position of the active enemy target.
            timestamp: Frame timestamp.
            target_id: Optional track ID of target.

        Returns:
            True if target timed out (>2.5s motionless) and was blacklisted; False otherwise.
        """
        self.clean_expired(timestamp)

        if target_pos is None:
            self._enemy_stationary = None
            return False

        if self._enemy_stationary is None:
            self._enemy_stationary = TargetStationaryState(
                target_id=target_id,
                last_position=target_pos,
                stationary_since=timestamp,
            )
            return False

        # Check movement distance against V1 threshold (20px)
        dist = target_pos.distance_to(self._enemy_stationary.last_position)
        if dist > self.CORPSE_STATIONARY_RADIUS_PX:
            # Target moved -> reset timer to current position
            self._enemy_stationary = TargetStationaryState(
                target_id=target_id,
                last_position=target_pos,
                stationary_since=timestamp,
            )
            return False

        # Motionless duration check (>2.5s)
        if (timestamp - self._enemy_stationary.stationary_since) > self.CORPSE_STATIONARY_TIMEOUT_SEC:
            # Blacklist this zone for 20 seconds!
            zone = SuppressionZone(
                position=target_pos,
                radius_px=self.CORPSE_ZONE_RADIUS_PX,
                expiry=timestamp + self.CORPSE_SUPPRESSION_DURATION_SEC,
                reason="CORPSE"
            )
            self._suppression_zones.append(zone)
            self._enemy_stationary = None
            return True

        return False

    def update_creep_attack(
        self,
        creep_pos: Optional[Vector2],
        is_attacking: bool,
        timestamp: float,
        target_id: Optional[int] = None
    ) -> bool:
        """
        ALLY-CREEP WATCHDOG:
        Monitors whether a farm target is attacked continuously for >4.5s.
        If so, flags it as invulnerable / allied creep and blacklists its location for 20s.

        Args:
            creep_pos: Position of attacked creep.
            is_attacking: True if our hero is currently attacking this creep.
            timestamp: Frame timestamp.
            target_id: Optional track ID of creep.

        Returns:
            True if creep timed out (>4.5s under attack) and was suppressed; False otherwise.
        """
        self.clean_expired(timestamp)

        if creep_pos is None or not is_attacking:
            self._creep_attack = None
            return False

        if self._creep_attack is None:
            self._creep_attack = AttackEngagementState(
                target_id=target_id,
                attack_started_at=timestamp,
                last_attacked_at=timestamp,
            )
            return False

        self._creep_attack.last_attacked_at = timestamp

        # Check continuous attack duration against V1 threshold (4.5s)
        if (timestamp - self._creep_attack.attack_started_at) > self.ALLY_CREEP_ATTACK_TIMEOUT_SEC:
            # Blacklist this zone for 20 seconds!
            zone = SuppressionZone(
                position=creep_pos,
                radius_px=self.ALLY_CREEP_ZONE_RADIUS_PX,
                expiry=timestamp + self.ALLY_CREEP_SUPPRESSION_DURATION_SEC,
                reason="INVULNERABLE_CREEP"
            )
            self._suppression_zones.append(zone)
            self._creep_attack = None
            return True

        return False

    def filter_suppressed_tracks(self, tracks: List[Track]) -> List[Track]:
        """
        Returns only tracks whose positions are not inside active suppression zones.
        """
        return [t for t in tracks if not self.is_suppressed(t.position)]

    def get_active_suppression_zones(self) -> List[SuppressionZone]:
        return list(self._suppression_zones)

    def reset(self) -> None:
        """Clears all suppression zones and watchdog state."""
        self._suppression_zones.clear()
        self._enemy_stationary = None
        self._creep_attack = None
