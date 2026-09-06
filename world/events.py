"""
world.events — Watchdogs, spatial suppression zones, and deterministic game events.

Provides:
  1. Anti-Corpse Watchdog (flags stationary targets >2.5s as dead/textures).
  2. Ally-Creep Watchdog (flags invulnerable creeps under continuous attack >4.5s).
  3. Spatial suppression zones (corpse / ally blacklist with 20.0s expiration).
  4. GameEvent models & WorldEventDetector for reliable events (HERO_KILL, CREEP_KILL,
     PLAYER_DEATH, TARGET_ENTERED, TARGET_LOST) with deduplication.

Strictly deterministic: uses explicit timestamps passed in update().
Does NOT decide tactical actions, does NOT grant rewards, and does NOT
declare hero death simply based on track loss.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Set
import math

from world.models import Vector2, WorldState
from world.tracker import Track, TrackLifecycle


class EventType(str, Enum):
    HERO_KILL = "hero_kill"
    CREEP_KILL = "creep_kill"
    PLAYER_DEATH = "player_death"
    TARGET_ENTERED = "target_entered"
    TARGET_LOST = "target_lost"


@dataclass(frozen=True)
class GameEvent:
    """
    Deterministic domain event on the battlefield.
    """
    type: EventType
    timestamp: float
    entity_id: Optional[int]
    confidence: float
    evidence: Tuple[str, ...]


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


class WorldEventDetector:
    """
    Detects and confirms high-level domain events from WorldState, active tracks,
    and combat history. Prevents false positive kill confirmations and deduplicates events.
    """

    # V1 Kill Confirmation Thresholds:
    MIN_COMBAT_DURATION_SEC = 2.0
    MAX_ATTACK_FRESHNESS_SEC = 3.5
    ENEMY_DISAPPEAR_TIMEOUT_SEC = 2.5

    # V1 Creep Kill Confirmation Thresholds:
    MIN_CREEP_ATTACK_DURATION_SEC = 2.5
    CREEP_DISAPPEAR_TIMEOUT_SEC = 2.0

    # V1 Player Death Thresholds:
    PLAYER_DISAPPEAR_TIMEOUT_SEC = 2.5
    PLAYER_LOW_HP_THRESHOLD = 0.28
    RECENT_COMBAT_TIMEOUT_SEC = 4.0

    def __init__(self, watchdog_manager: Optional[WatchdogManager] = None):
        self.watchdog = watchdog_manager or WatchdogManager()

        # Observation tracking for TARGET_ENTERED / TARGET_LOST
        self._previous_track_ids: Set[int] = set()

        # Combat tracking state
        self._combat_start_time: Optional[float] = None
        self._last_combat_attack_time: Optional[float] = None
        self._last_attacked_enemy_pos: Optional[Vector2] = None
        self._enemy_disappeared_since: Optional[float] = None

        # Creep farming state
        self._creep_attack_start_time: Optional[float] = None
        self._last_farm_pos: Optional[Vector2] = None
        self._creep_disappeared_since: Optional[float] = None

        # Player state
        self._last_player_seen_time: float = 0.0
        self._last_known_hp_ratio: float = 1.0
        self._is_player_dead: bool = False
        self._player_death_time: float = 0.0

        # Event deduplication
        self._confirmed_kills_count: int = 0
        self._killed_hero_positions: List[Tuple[Vector2, float]] = []

    def record_attack_action(self, target_pos: Vector2, timestamp: float, is_hero: bool = True) -> None:
        """
        Informs detector that our hero launched an attack at target_pos.
        Provides the ground truth for engagement duration and recency.
        """
        if is_hero:
            if self._combat_start_time is None:
                self._combat_start_time = timestamp
            self._last_combat_attack_time = timestamp
            self._last_attacked_enemy_pos = target_pos
        else:
            if self._creep_attack_start_time is None:
                self._creep_attack_start_time = timestamp
            self._last_farm_pos = target_pos

    def update(
        self,
        world_state: WorldState,
        tracks: List[Track],
        timestamp: float,
        turret_danger: bool = False,
    ) -> List[GameEvent]:
        """
        Updates event detection logic and returns list of newly fired events.
        """
        events: List[GameEvent] = []

        # 1. Track Observation Events: TARGET_ENTERED / TARGET_LOST
        current_track_ids = {t.track_id for t in tracks if t.visible}
        new_ids = current_track_ids - self._previous_track_ids
        lost_ids = self._previous_track_ids - current_track_ids

        for tid in new_ids:
            events.append(GameEvent(
                type=EventType.TARGET_ENTERED,
                timestamp=timestamp,
                entity_id=tid,
                confidence=1.0,
                evidence=("track_appeared",)
            ))

        for tid in lost_ids:
            events.append(GameEvent(
                type=EventType.TARGET_LOST,
                timestamp=timestamp,
                entity_id=tid,
                confidence=1.0,
                evidence=("track_lost_from_frame",)
            ))

        self._previous_track_ids = current_track_ids

        # 2. Player Death Detection
        player = world_state.player
        if player and player.is_visible:
            self._last_player_seen_time = timestamp
            self._last_known_hp_ratio = player.hp.value
            if self._is_player_dead:
                self._is_player_dead = False
        else:
            in_combat_recently = (
                self._last_combat_attack_time is not None and
                (timestamp - self._last_combat_attack_time) < self.RECENT_COMBAT_TIMEOUT_SEC
            ) or turret_danger

            if (timestamp - self._last_player_seen_time > self.PLAYER_DISAPPEAR_TIMEOUT_SEC) and not self._is_player_dead:
                if in_combat_recently and self._last_known_hp_ratio < self.PLAYER_LOW_HP_THRESHOLD:
                    self._is_player_dead = True
                    self._player_death_time = timestamp
                    events.append(GameEvent(
                        type=EventType.PLAYER_DEATH,
                        timestamp=timestamp,
                        entity_id=None,
                        confidence=0.95,
                        evidence=(
                            f"low_hp_{self._last_known_hp_ratio:.2f}",
                            "missing_during_combat",
                        )
                    ))

        # 3. Enemy Hero Kill Confirmation (V1 Anti-Corpse / Combat Duration Criteria)
        visible_enemies = [t for t in tracks if t.visible and "enemy" in t.class_name]

        if visible_enemies:
            self._enemy_disappeared_since = None
        else:
            if self._last_attacked_enemy_pos is not None:
                if self._enemy_disappeared_since is None:
                    self._enemy_disappeared_since = timestamp
                elif (timestamp - self._enemy_disappeared_since) >= self.ENEMY_DISAPPEAR_TIMEOUT_SEC:
                    combat_duration = (
                        (self._last_combat_attack_time - self._combat_start_time)
                        if (self._last_combat_attack_time and self._combat_start_time) else 0.0
                    )
                    attack_freshness = (
                        (timestamp - self._last_combat_attack_time)
                        if self._last_combat_attack_time else 999.0
                    )

                    # Strict V1 kill criteria:
                    # Combat lasted >= 2.0s AND last attack was fresh (<= 3.5s) AND disappeared >= 2.5s
                    if combat_duration >= self.MIN_COMBAT_DURATION_SEC and attack_freshness <= self.MAX_ATTACK_FRESHNESS_SEC:
                        events.append(GameEvent(
                            type=EventType.HERO_KILL,
                            timestamp=timestamp,
                            entity_id=None,
                            confidence=0.95,
                            evidence=(
                                f"combat_duration_{combat_duration:.1f}s",
                                f"attack_freshness_{attack_freshness:.1f}s",
                                f"disappeared_{self.ENEMY_DISAPPEAR_TIMEOUT_SEC}s",
                            )
                        ))
                        self._confirmed_kills_count += 1

                    # Reset combat state after kill confirmation or timeout
                    self._last_attacked_enemy_pos = None
                    self._combat_start_time = None
                    self._last_combat_attack_time = None
                    self._enemy_disappeared_since = None

        # 4. Creep Kill Confirmation
        # Check if creep disappeared after >= 2.5s of continuous attack
        visible_minions = [t for t in tracks if t.visible and ("minion" in t.class_name or "buff" in t.class_name)]
        if visible_minions:
            self._creep_disappeared_since = None
        else:
            if self._last_farm_pos is not None:
                if self._creep_disappeared_since is None:
                    self._creep_disappeared_since = timestamp
                elif (timestamp - self._creep_disappeared_since) >= self.CREEP_DISAPPEAR_TIMEOUT_SEC:
                    attack_duration = (
                        (timestamp - self._creep_attack_start_time)
                        if self._creep_attack_start_time else 0.0
                    )
                    if attack_duration >= self.MIN_CREEP_ATTACK_DURATION_SEC:
                        # Ensure not an invulnerable/ally suppressed creep
                        if not self.watchdog.is_suppressed(self._last_farm_pos):
                            events.append(GameEvent(
                                type=EventType.CREEP_KILL,
                                timestamp=timestamp,
                                entity_id=None,
                                confidence=0.90,
                                evidence=(
                                    f"attack_duration_{attack_duration:.1f}s",
                                    f"disappeared_{self.CREEP_DISAPPEAR_TIMEOUT_SEC}s",
                                )
                            ))

                    self._last_farm_pos = None
                    self._creep_attack_start_time = None
                    self._creep_disappeared_since = None

        return events

    def reset(self) -> None:
        """Clears detector state and history."""
        self.watchdog.reset()
        self._previous_track_ids.clear()
        self._combat_start_time = None
        self._last_combat_attack_time = None
        self._last_attacked_enemy_pos = None
        self._enemy_disappeared_since = None
        self._creep_attack_start_time = None
        self._last_farm_pos = None
        self._creep_disappeared_since = None
        self._last_player_seen_time = 0.0
        self._last_known_hp_ratio = 1.0
        self._is_player_dead = False
        self._player_death_time = 0.0
        self._confirmed_kills_count = 0
