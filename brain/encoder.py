"""Tactical state encoder for Claude Tactical AI V2.

Encodes an observed TacticalState into a canonical discrete state_key string
for Q-table lookups and reinforcement learning updates.

Preserves exact 1:1 V1 baseline discretization for all three Q-domains:
- Combat: {dist_bucket}_{hp_cat}_COMBO_{can_combo} (plus overrides TURRET_DANGER, CRITICAL_HP_DANGER)
- Farm: {dist_bucket}_S1_{s1_ok}_{hp_cat}
- Roam: SCOUT_HOT_ZONE / SCOUT_MID_SEARCH / SCOUT_DEEP_PATROL
"""

from __future__ import annotations

import math
from typing import Optional, Union

from brain.state import TacticalState


class TacticalStateEncoder:
    """Deterministic encoder translating continuous/discrete TacticalState into Q-table state keys."""

    # Discretization thresholds extracted 1:1 from V1 claude_brain.py
    COMBAT_DANGER_CLOSE_MAX: float = 260.0
    COMBAT_SWEET_SPOT_MAX: float = 430.0
    COMBAT_HP_HIGH_MIN: float = 0.65
    COMBAT_HP_CRITICAL_MAX: float = 0.32

    FARM_DANGER_CLOSE_MAX: float = 260.0
    FARM_FAR_MIN: float = 420.0
    FARM_HP_SAFE_MIN: float = 0.50

    ROAM_HOT_ZONE_MAX: float = 5.0
    ROAM_MID_SEARCH_MAX: float = 15.0

    def _validate_state(self, state: TacticalState) -> None:
        """Validates that state fields are structurally sound and free of NaN/Inf."""
        if not isinstance(state, TacticalState):
            raise TypeError(f"Expected TacticalState instance, got: {type(state).__name__}")

        # Check scalar numeric fields
        for name, val in (
            ("timestamp", state.timestamp),
            ("player_hp_ratio", state.player_hp_ratio),
        ):
            if val is None or not isinstance(val, (int, float)):
                raise TypeError(f"Field '{name}' must be a numeric float/int")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Field '{name}' cannot be NaN or Inf (got {val})")

        # Check player position
        if state.player_position is None:
            raise ValueError("Field 'player_position' cannot be None")
        for coord_name, coord in (("x", state.player_position.x), ("y", state.player_position.y)):
            if math.isnan(coord) or math.isinf(coord):
                raise ValueError(f"player_position.{coord_name} cannot be NaN or Inf")

        # Check optional enemy distance and position
        if state.nearest_enemy_distance is not None:
            if math.isnan(state.nearest_enemy_distance) or math.isinf(state.nearest_enemy_distance):
                raise ValueError(f"nearest_enemy_distance cannot be NaN or Inf (got {state.nearest_enemy_distance})")
            if state.nearest_enemy_distance < 0.0:
                raise ValueError(f"nearest_enemy_distance cannot be negative (got {state.nearest_enemy_distance})")

        if state.nearest_enemy_position is not None:
            for c_name, c_val in (("x", state.nearest_enemy_position.x), ("y", state.nearest_enemy_position.y)):
                if math.isnan(c_val) or math.isinf(c_val):
                    raise ValueError(f"nearest_enemy_position.{c_name} cannot be NaN or Inf")

        # Check optional minion distance and position
        if state.nearest_minion_distance is not None:
            if math.isnan(state.nearest_minion_distance) or math.isinf(state.nearest_minion_distance):
                raise ValueError(f"nearest_minion_distance cannot be NaN or Inf (got {state.nearest_minion_distance})")
            if state.nearest_minion_distance < 0.0:
                raise ValueError(f"nearest_minion_distance cannot be negative (got {state.nearest_minion_distance})")

        if state.nearest_minion_position is not None:
            for c_name, c_val in (("x", state.nearest_minion_position.x), ("y", state.nearest_minion_position.y)):
                if math.isnan(c_val) or math.isinf(c_val):
                    raise ValueError(f"nearest_minion_position.{c_name} cannot be NaN or Inf")

    def encode_combat(self, state: TacticalState) -> str:
        """Encodes state into combat_q key using exact V1 baseline rules."""
        self._validate_state(state)

        # 1. Hard safety overrides
        if state.enemy_near_turret:
            return "TURRET_DANGER"

        if state.player_hp_ratio < self.COMBAT_HP_CRITICAL_MAX:
            return "CRITICAL_HP_DANGER"

        # 2. Distance discretization
        dist = state.nearest_enemy_distance if state.nearest_enemy_distance is not None else 300.0
        if dist < self.COMBAT_DANGER_CLOSE_MAX:
            dist_bucket = "DANGER_CLOSE"
        elif dist <= self.COMBAT_SWEET_SPOT_MAX:
            dist_bucket = "SWEET_SPOT"
        else:
            dist_bucket = "CHASE_FAR"

        # 3. Health category
        hp_cat = "HIGH" if state.player_hp_ratio >= self.COMBAT_HP_HIGH_MIN else "MID"

        # 4. Combo readiness
        can_combo = bool(state.can_combo)

        return f"{dist_bucket}_{hp_cat}_COMBO_{can_combo}"

    def encode_farm(self, state: TacticalState) -> str:
        """Encodes state into farm_q key using exact V1 baseline rules."""
        self._validate_state(state)

        # 1. Distance discretization
        dist = state.nearest_minion_distance if state.nearest_minion_distance is not None else 300.0
        if dist > self.FARM_FAR_MIN:
            dist_bucket = "CREEP_FAR"
        elif dist < self.FARM_DANGER_CLOSE_MAX:
            dist_bucket = "CREEP_DANGER_CLOSE"
        else:
            dist_bucket = "CREEP_SWEET_SPOT"

        # 2. Skill readiness and HP category
        s1_ok = bool(state.s1_ready)
        hp_cat = "SAFE" if state.player_hp_ratio >= self.FARM_HP_SAFE_MIN else "LOW"

        return f"{dist_bucket}_S1_{s1_ok}_{hp_cat}"

    def encode_roam(self, state: TacticalState, last_enemy_seen_time: float = 0.0) -> str:
        """Encodes state into roam_q key using exact V1 baseline rules."""
        self._validate_state(state)
        time_since_enemy = max(0.0, state.timestamp - float(last_enemy_seen_time))

        if time_since_enemy < self.ROAM_HOT_ZONE_MAX:
            return "SCOUT_HOT_ZONE"
        if time_since_enemy < self.ROAM_MID_SEARCH_MAX:
            return "SCOUT_MID_SEARCH"
        return "SCOUT_DEEP_PATROL"

    def encode(
        self,
        state: TacticalState,
        context: Optional[str] = None,
        last_enemy_seen_time: float = 0.0,
    ) -> str:
        """Encodes state for the specified context or automatically detects operational mode."""
        self._validate_state(state)

        target_context = str(context).lower().strip() if context is not None else None

        if target_context in ("combat", "combat_q"):
            return self.encode_combat(state)

        if target_context in ("farm", "farm_q"):
            return self.encode_farm(state)

        if target_context in ("roam", "roam_q"):
            return self.encode_roam(state, last_enemy_seen_time=last_enemy_seen_time)

        # Automatic detection matching ClaudeBrainV2 branching
        if state.enemy_visible and state.nearest_enemy_distance is not None:
            return self.encode_combat(state)

        if state.minion_count > 0 and state.nearest_minion_distance is not None:
            return self.encode_farm(state)

        return self.encode_roam(state, last_enemy_seen_time=last_enemy_seen_time)
