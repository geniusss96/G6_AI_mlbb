"""
brain.v1_adapter — Adapter bridging Claude Tactical AI V2 contracts to V1 ClaudeRLBrain.

Contract:
    WorldState → V1BrainAdapter.decide() → Action

Translates V2 observed WorldState into legacy method arguments:
    evaluate_combat_stance(dist_enemy, can_combo, is_turret_near, hp_self_ratio)
    evaluate_farm_stance(dist_farm, s1_ok, hp_self_ratio)
    choose_roam_direction(ref_pt, time_since_enemy)

And maps legacy decisions ("KITE_AND_POKE", "TURRET_RETREAT", "FARM_APPROACH", etc.)
into typed Action objects.

Strictly isolated from ADB, subprocess, joystick, and optical frame processing.
"""

from typing import Optional, Tuple, Any, List
import math

from world.models import Vector2, WorldState, EnemyState, MinionState, TurretState
from actions.models import Action, ActionType


class V1BrainAdapter:
    """
    Adapter encapsulating a legacy ClaudeRLBrain instance.
    Implements V2 Tactical Brain contract: WorldState -> Action.
    """

    def __init__(self, legacy_brain: Any):
        if legacy_brain is None:
            raise ValueError("legacy_brain cannot be None")
        self.legacy_brain = legacy_brain
        self.last_enemy_seen_time: float = 0.0

    def decide(self, world_state: WorldState) -> Action:
        """
        Converts WorldState to legacy input parameters, queries legacy brain,
        and translates the resulting decision into a typed Action.
        """
        if not isinstance(world_state, WorldState):
            raise TypeError(f"V1BrainAdapter expects WorldState, got: {type(world_state)}")

        # If player is marked invisible / dead, return IDLE
        if world_state.player and not world_state.player.is_visible:
            return Action(type=ActionType.IDLE)

        # 1. Extract player reference position and HP ratio
        player_pos = world_state.player.position if world_state.player else Vector2(772.0, 360.0)
        hp_ratio = float(world_state.player.hp.value) if (world_state.player and world_state.player.hp) else 1.0

        # 2. Extract skill readiness
        s1_ready = world_state.skills.s1_ready if world_state.skills else True
        s2_ready = world_state.skills.s2_ready if world_state.skills else True
        ult_ready = world_state.skills.ultimate_ready if world_state.skills else True
        can_combo = s2_ready and ult_ready

        # ----------------------------------------------------------------------
        # Branch 1: Enemy hero present -> Combat mode
        # ----------------------------------------------------------------------
        if world_state.enemies:
            self.last_enemy_seen_time = world_state.timestamp
            closest_enemy = self._find_closest_enemy(player_pos, world_state.enemies)
            dist_enemy = player_pos.distance_to(closest_enemy.position)

            # Check if any enemy turret is within danger distance (420px)
            is_turret_near = any(
                player_pos.distance_to(t.position) < 420.0 and t.is_enemy
                for t in world_state.turrets
            )

            legacy_decision = self.legacy_brain.evaluate_combat_stance(
                dist_enemy=dist_enemy,
                can_combo=can_combo,
                is_turret_near=is_turret_near,
                hp_self_ratio=hp_ratio,
            )

            return self._from_legacy_combat_decision(
                decision=legacy_decision,
                player_pos=player_pos,
                enemy=closest_enemy,
                dist_enemy=dist_enemy,
            )

        # ----------------------------------------------------------------------
        # Branch 2: No enemies, but minions/creeps present -> Farm mode
        # ----------------------------------------------------------------------
        enemy_minions = [m for m in world_state.minions if m.is_enemy]
        if enemy_minions:
            closest_minion = self._find_closest_minion(player_pos, enemy_minions)
            dist_farm = player_pos.distance_to(closest_minion.position)

            legacy_decision = self.legacy_brain.evaluate_farm_stance(
                dist_farm=dist_farm,
                s1_ok=s1_ready,
                hp_self_ratio=hp_ratio,
            )

            return self._from_legacy_farm_decision(
                decision=legacy_decision,
                player_pos=player_pos,
                minion=closest_minion,
            )

        # ----------------------------------------------------------------------
        # Branch 3: No enemies, no creeps -> Exploration / Roam mode
        # ----------------------------------------------------------------------
        time_since_enemy = max(0.0, world_state.timestamp - self.last_enemy_seen_time)
        res = self.legacy_brain.choose_roam_direction(
            ref_pt=player_pos.as_tuple(),
            time_since_enemy=time_since_enemy,
        )

        if isinstance(res, tuple) and len(res) == 2:
            chosen_action, target_pt = res
            return Action(
                type=ActionType.SCOUT,
                direction=Vector2(float(target_pt[0]), float(target_pt[1])),
            )

        return Action(type=ActionType.IDLE)

    def _find_closest_enemy(self, player_pos: Vector2, enemies: List[EnemyState]) -> EnemyState:
        return min(enemies, key=lambda e: player_pos.distance_to(e.position))

    def _find_closest_minion(self, player_pos: Vector2, minions: List[MinionState]) -> MinionState:
        return min(minions, key=lambda m: player_pos.distance_to(m.position))

    def _from_legacy_combat_decision(
        self,
        decision: str,
        player_pos: Vector2,
        enemy: EnemyState,
        dist_enemy: float,
    ) -> Action:
        """Converts legacy combat decisions into typed Action models."""
        target_id = getattr(enemy, "track_id", None)

        if decision == "TURRET_RETREAT":
            # Retreat away from turret / back towards home base
            safe_point = Vector2(player_pos.x - 180.0, player_pos.y + 90.0)
            return Action(type=ActionType.RETREAT, direction=safe_point, target_id=target_id)

        if decision == "TACTICAL_RETREAT":
            safe_point = Vector2(player_pos.x - 180.0, player_pos.y + 90.0)
            return Action(type=ActionType.RETREAT, direction=safe_point, target_id=target_id)

        if decision == "KITE_AND_POKE":
            return Action(type=ActionType.KITE, direction=enemy.position, target_id=target_id)

        if decision == "SWEET_SPOT_BURST":
            # Orbital strafe around enemy
            dx = enemy.position.x - player_pos.x
            dy = enemy.position.y - player_pos.y
            strafe_pos = Vector2(player_pos.x - dy * 0.45, player_pos.y + dx * 0.45)
            return Action(type=ActionType.ATTACK, direction=strafe_pos, target_id=target_id)

        if decision == "DIVE_ALL_IN":
            return Action(type=ActionType.CAST_ULT, direction=enemy.position, target_id=target_id)

        # Fallback for unexpected string decisions
        return Action(type=ActionType.IDLE)

    def _from_legacy_farm_decision(
        self,
        decision: str,
        player_pos: Vector2,
        minion: MinionState,
    ) -> Action:
        """Converts legacy farm decisions into typed Action models."""
        target_id = getattr(minion, "track_id", None)

        if decision == "FARM_APPROACH":
            return Action(type=ActionType.MOVE, direction=minion.position, target_id=target_id)

        if decision == "FARM_KITE_BACK":
            return Action(type=ActionType.KITE, direction=minion.position, target_id=target_id)

        if decision == "FARM_S1_AOE":
            return Action(type=ActionType.CAST_S1, target_id=target_id)

        if decision == "FARM_SWEET_SPOT":
            # Orbital orb-walk around creep
            dx = minion.position.x - player_pos.x
            dy = minion.position.y - player_pos.y
            strafe_pos = Vector2(player_pos.x - dy * 0.40, player_pos.y + dx * 0.40)
            return Action(type=ActionType.FARM, direction=strafe_pos, target_id=target_id)

        return Action(type=ActionType.IDLE)
