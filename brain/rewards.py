"""
brain.rewards — Centralized reward models and calculation engine for Claude Tactical AI V2.

Extracted 1:1 from V1 ClaudeRLBrain and realtime_vision.py reward baseline.
Calculates deterministic numerical reinforcement signals for game events and tactical outcomes:
- Hero kills / deaths
- Creep last-hits
- Scout discoveries
- Combat engagement & retreat penalties
- Wall collisions & idle exploration penalties

Pure evaluation layer: does NOT mutate Q-tables, does NOT perform Experience Replay,
and does NOT trigger file I/O.
"""

from dataclasses import dataclass
from typing import List, Optional

from world.events import EventType, GameEvent


@dataclass(frozen=True)
class Reward:
    """
    Immutable representation of a numerical reinforcement signal.

    Attributes:
        value: Exact numeric reward or penalty (e.g. +60.0, -40.0).
        reason: Human-readable diagnostic description from V1 baseline.
        source: Trigger identifier ("hero_kill", "tactical", etc.).
        domain: Target Q-table domain ("combat", "farm", "roam").
    """
    value: float
    reason: str
    source: str
    domain: str = "combat"


class RewardCalculator:
    """
    Pure calculator translating GameEvents and tactical context into discrete Rewards.
    Preserves exact V1 baseline numerical values without scaling or clipping.
    """

    # V1 Baseline constants
    REWARD_HERO_KILL_COMBAT = 60.0
    REWARD_HERO_KILL_ROAM = 40.0
    PENALTY_PLAYER_DEATH_COMBAT = -40.0
    PENALTY_PLAYER_DEATH_ROAM = -30.0
    REWARD_CREEP_KILL_FARM = 25.0
    REWARD_CREEP_KILL_ROAM = 15.0
    REWARD_ENEMY_SPOTTED_ROAM = 30.0
    PENALTY_TURRET_WALK_COMBAT = -40.0
    REWARD_BURST_COMBO_COMBAT = 35.0
    REWARD_S1_AOE_FARM = 20.0
    PENALTY_WALL_COLLISION_ROAM = -25.0
    PENALTY_IDLE_ROAM = -3.0

    def for_event(self, event: GameEvent) -> List[Reward]:
        """
        Maps a discrete GameEvent to corresponding V1 rewards across affected domains.
        """
        if not isinstance(event, GameEvent):
            raise TypeError(f"RewardCalculator expects GameEvent, got: {type(event)}")

        if event.type == EventType.HERO_KILL:
            return [
                Reward(
                    value=self.REWARD_HERO_KILL_COMBAT,
                    reason="KILLED ENEMY HERO",
                    source="hero_kill",
                    domain="combat",
                ),
                Reward(
                    value=self.REWARD_HERO_KILL_ROAM,
                    reason="ROUTE LED TO KILL",
                    source="hero_kill",
                    domain="roam",
                ),
            ]

        if event.type == EventType.PLAYER_DEATH:
            return [
                Reward(
                    value=self.PENALTY_PLAYER_DEATH_COMBAT,
                    reason="DIED IN COMBAT",
                    source="player_death",
                    domain="combat",
                ),
                Reward(
                    value=self.PENALTY_PLAYER_DEATH_ROAM,
                    reason="DIED ON ROUTE",
                    source="player_death",
                    domain="roam",
                ),
            ]

        if event.type == EventType.CREEP_KILL:
            return [
                Reward(
                    value=self.REWARD_CREEP_KILL_FARM,
                    reason="CREEP SLAIN (LAST HIT)",
                    source="creep_kill",
                    domain="farm",
                ),
                Reward(
                    value=self.REWARD_CREEP_KILL_ROAM,
                    reason="FOUND JUNGLE/LANE FARM",
                    source="creep_kill",
                    domain="roam",
                ),
            ]

        if event.type == EventType.TARGET_ENTERED:
            return [
                Reward(
                    value=self.REWARD_ENEMY_SPOTTED_ROAM,
                    reason="FOUND ENEMY HERO",
                    source="target_entered",
                    domain="roam",
                )
            ]

        return []

    # Contextual tactical rewards matching V1 realtime_vision.py branches

    def for_turret_danger(self) -> Reward:
        """Penalty for moving into enemy turret range."""
        return Reward(
            value=self.PENALTY_TURRET_WALK_COMBAT,
            reason="WALKED INTO TURRET",
            source="tactical",
            domain="combat",
        )

    def for_burst_combo(self) -> Reward:
        """Reward for executing full Claude all-in burst combo."""
        return Reward(
            value=self.REWARD_BURST_COMBO_COMBAT,
            reason="BURST COMBO EXECUTED",
            source="tactical",
            domain="combat",
        )

    def for_s1_aoe_wave(self) -> Reward:
        """Reward for clearing minion cluster with S1 AOE."""
        return Reward(
            value=self.REWARD_S1_AOE_FARM,
            reason="S1 AOE CAST ON WAVE",
            source="tactical",
            domain="farm",
        )

    def for_wall_collision(self) -> Reward:
        """Penalty for getting stuck in geometry/obstacles."""
        return Reward(
            value=self.PENALTY_WALL_COLLISION_ROAM,
            reason="WALL COLLISION",
            source="tactical",
            domain="roam",
        )

    def for_idle_penalty(self) -> Reward:
        """Soft penalty for wandering without combat/farm contact for >25s."""
        return Reward(
            value=self.PENALTY_IDLE_ROAM,
            reason="EMPTY PATH (IDLE)",
            source="tactical",
            domain="roam",
        )

    def for_custom(
        self,
        value: float,
        reason: str,
        domain: str = "combat",
        source: str = "custom",
    ) -> Reward:
        """Arbitrary reward wrapper preserving value semantics."""
        return Reward(value=float(value), reason=reason, source=source, domain=domain)
