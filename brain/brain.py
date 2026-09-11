"""Claude Tactical AI V2 — Brain Composition.

Assembles the extracted components (TacticalState, QLearningCore, RewardCalculator,
BrainMemory, and Action models) into ClaudeBrainV2.

Operates purely on abstract contracts:
    TacticalState -> ClaudeBrainV2.decide() -> Action

Zero awareness of ADB, Android touchscreen coordinates, joystick driver,
or optical frame processing.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from actions.models import Action, ActionType
from brain.memory import BrainMemory
from brain.q_learning import QLearningCore
from brain.rewards import Reward, RewardCalculator
from brain.state import TacticalState
from world.models import Vector2


class ClaudeBrainV2:
    """Tactical Brain engine for V2 architecture.

    Coordinates Q-learning policy evaluation, tactical state discretization,
    action synthesis, and memory tracking via dependency injection.
    """

    DUEL_ACTIONS: List[str] = [
        "KITE_AND_POKE",
        "SWEET_SPOT_BURST",
        "DIVE_ALL_IN",
    ]

    FARM_ACTIONS: List[str] = [
        "FARM_APPROACH",
        "FARM_KITE_BACK",
        "FARM_SWEET_SPOT",
        "FARM_S1_AOE",
    ]

    ROAM_ACTIONS: Dict[str, Tuple[float, float]] = {
        "LANE_ADVANCE": (230.0, -60.0),
        "LANE_PUSH": (190.0, -20.0),
        "RIVER_SCOUT": (130.0, 90.0),
        "FLANK_ADVANCE": (110.0, -130.0),
        "CREEP_INTERCEPT": (160.0, 40.0),
    }

    ROAM_OFFSETS = ROAM_ACTIONS

    DECISION_TO_ACTION_TYPE: Dict[str, ActionType] = {
        "TURRET_RETREAT": ActionType.RETREAT,
        "TACTICAL_RETREAT": ActionType.RETREAT,
        "KITE_AND_POKE": ActionType.KITE,
        "SWEET_SPOT_BURST": ActionType.ATTACK,
        "DIVE_ALL_IN": ActionType.CAST_ULT,
        "FARM_APPROACH": ActionType.MOVE,
        "FARM_KITE_BACK": ActionType.KITE,
        "FARM_SWEET_SPOT": ActionType.FARM,
        "FARM_S1_AOE": ActionType.CAST_S1,
        "LANE_ADVANCE": ActionType.MOVE,
        "LANE_PUSH": ActionType.MOVE,
        "RIVER_SCOUT": ActionType.SCOUT,
        "FLANK_ADVANCE": ActionType.SCOUT,
        "CREEP_INTERCEPT": ActionType.FARM,
    }

    def __init__(
        self,
        q_learning: Optional[QLearningCore] = None,
        rewards: Optional[RewardCalculator] = None,
        memory: Optional[BrainMemory] = None,
        tactical_policy: Optional[Any] = None,
        random_generator: Optional[Any] = None,
    ) -> None:
        self.q_learning: QLearningCore = (
            q_learning if q_learning is not None else QLearningCore()
        )
        self.rewards: RewardCalculator = (
            rewards if rewards is not None else RewardCalculator()
        )
        self.memory: BrainMemory = (
            memory if memory is not None else BrainMemory()
        )
        self.tactical_policy: Optional[Any] = tactical_policy
        self._rng: Any = (
            random_generator if random_generator is not None else random
        )

        # Runtime tactical history
        self.last_enemy_seen_time: float = 0.0
        self.last_decision: Optional[str] = None

        self.last_combat_state: Optional[str] = None
        self.last_combat_action: Optional[str] = None

        self.last_farm_state: Optional[str] = None
        self.last_farm_action: Optional[str] = None

        self.last_roam_state: Optional[str] = None
        self.last_roam_action: Optional[str] = None

        self.total_decisions: int = 0

    def decide(self, tactical_state: TacticalState) -> Action:
        """Evaluate tactical state and output the next abstract Action."""

        if not isinstance(tactical_state, TacticalState):
            raise TypeError(
                f"decide expects TacticalState, got: {type(tactical_state)}"
            )

        # ------------------------------------------------------------------
        # Player dead -> idle
        # ------------------------------------------------------------------
        if tactical_state.is_player_dead:
            return Action(type=ActionType.IDLE)

        # ------------------------------------------------------------------
        # Branch 1: Enemy Hero in sight -> Combat
        # ------------------------------------------------------------------
        if (
            tactical_state.enemy_visible
            and tactical_state.nearest_enemy_distance is not None
        ):
            self.last_enemy_seen_time = tactical_state.timestamp

            decision = self._decide_combat(tactical_state)

            self.last_decision = decision
            self.total_decisions += 1

            if self.memory:
                self.memory.log_action(
                    category="COMBAT",
                    state=self.last_combat_state or "COMBAT",
                    action=decision,
                    timestamp=tactical_state.timestamp,
                )

            return self.map_decision_to_action(
                decision,
                tactical_state,
            )

        # ------------------------------------------------------------------
        # Branch 2: Minions/Creeps in sight -> Farm
        # ------------------------------------------------------------------
        if (
            tactical_state.minion_count > 0
            and tactical_state.nearest_minion_distance is not None
        ):
            decision = self._decide_farm(tactical_state)

            self.last_decision = decision
            self.total_decisions += 1

            if self.memory:
                self.memory.log_action(
                    category="FARM",
                    state=self.last_farm_state or "FARM",
                    action=decision,
                    timestamp=tactical_state.timestamp,
                )

            return self.map_decision_to_action(
                decision,
                tactical_state,
            )

        # ------------------------------------------------------------------
        # Branch 3: Free lane -> Roam
        # ------------------------------------------------------------------
        decision = self._decide_roam(tactical_state)

        self.last_decision = decision
        self.total_decisions += 1

        if self.memory:
            self.memory.log_action(
                category="ROAM",
                state=self.last_roam_state or "ROAM",
                action=decision,
                timestamp=tactical_state.timestamp,
            )

        return self.map_decision_to_action(
            decision,
            tactical_state,
        )

    # ----------------------------------------------------------------------
    # Q-learning selection
    # ----------------------------------------------------------------------
    def _choose_epsilon_greedy(
        self,
        domain: str,
        state_key: str,
        actions: List[str],
        epsilon: float,
    ) -> str:
        """Execute epsilon-greedy choice through QLearningCore."""

        if not actions:
            raise ValueError(
                f"No actions available for domain={domain}, state={state_key}"
            )

        self.q_learning.ensure_state(
            table_name=domain,
            state_key=state_key,
            defaults={act: 0.0 for act in actions},
        )

        # Exploration
        if self._rng.random() < epsilon:
            return self._rng.choice(actions)

        # Exploitation
        chosen, _ = self.q_learning.get_best_action(
            table_name=domain,
            state_key=state_key,
            candidate_actions=actions,
            rng=self._rng,
        )

        return chosen

    # ----------------------------------------------------------------------
    # Combat decision
    # ----------------------------------------------------------------------
    def _decide_combat(self, ts: TacticalState) -> str:
        """Choose combat action using safety rules + Q-learning.

        Important:
        - Hard safety rules always win.
        - DIVE_ALL_IN is unavailable when combo is not ready.
        - Memory is only a weak hint.
        - Q-learning remains the primary decision source.
        """

        # ==============================================================
        # 1. HARD SAFETY OVERRIDES
        # ==============================================================

        if ts.enemy_near_turret:
            self.last_combat_state = "TURRET_DANGER"
            self.last_combat_action = "TURRET_RETREAT"
            return "TURRET_RETREAT"

        if ts.player_hp_ratio < 0.32:
            self.last_combat_state = "CRITICAL_HP"
            self.last_combat_action = "TACTICAL_RETREAT"
            return "TACTICAL_RETREAT"

        # ==============================================================
        # 2. DISCRETIZE CURRENT COMBAT STATE
        # ==============================================================

        dist = (
            ts.nearest_enemy_distance
            if ts.nearest_enemy_distance is not None
            else 300.0
        )

        if dist < 260.0:
            dist_cat = "DANGER_CLOSE"
        elif dist <= 430.0:
            dist_cat = "SWEET_SPOT"
        else:
            dist_cat = "CHASE_FAR"

        if ts.player_hp_ratio >= 0.65:
            hp_cat = "HIGH"
        else:
            hp_cat = "MID"

        state = (
            f"{dist_cat}_"
            f"{hp_cat}_"
            f"COMBO_{ts.can_combo}"
        )

        self.last_combat_state = state

        # ==============================================================
        # 3. BUILD ONLY LEGAL ACTIONS
        # ==============================================================

        actions = list(self.DUEL_ACTIONS)

        # Never allow all-in when combo is not ready.
        if not ts.can_combo and "DIVE_ALL_IN" in actions:
            actions.remove("DIVE_ALL_IN")

        # At reduced HP, also block all-in.
        if ts.player_hp_ratio < 0.45 and "DIVE_ALL_IN" in actions:
            actions.remove("DIVE_ALL_IN")

        # Very close + low HP -> prefer survival actions.
        if dist < 180.0 and ts.player_hp_ratio < 0.55:
            preferred = [
                action
                for action in (
                    "KITE_AND_POKE",
                    "SWEET_SPOT_BURST",
                )
                if action in actions
            ]

            if preferred:
                actions = preferred

        if not actions:
            actions = ["KITE_AND_POKE"]

        # ==============================================================
        # 4. Q-LEARNING IS THE PRIMARY DECISION MAKER
        # ==============================================================

        chosen_action = self._choose_epsilon_greedy(
            domain="combat",
            state_key=state,
            actions=actions,
            epsilon=0.08,
        )

        # ==============================================================
        # 5. MEMORY = WEAK OPTIONAL HINT
        # ==============================================================

        recalled = (
            self.memory.recall_best_tactic(actions)
            if self.memory
            else None
        )

        # Memory may influence only 15% of valid decisions.
        if (
            recalled in actions
            and self._rng.random() < 0.15
        ):
            chosen_action = recalled

        self.last_combat_action = chosen_action
        return chosen_action

    # ----------------------------------------------------------------------
    # Farm decision
    # ----------------------------------------------------------------------
    def _decide_farm(self, ts: TacticalState) -> str:
        dist = (
            ts.nearest_minion_distance
            if ts.nearest_minion_distance is not None
            else 300.0
        )

        if dist > 420.0:
            dist_cat = "CREEP_FAR"
        elif dist < 260.0:
            dist_cat = "CREEP_DANGER_CLOSE"
        else:
            dist_cat = "CREEP_SWEET_SPOT"

        hp_cat = (
            "SAFE"
            if ts.player_hp_ratio >= 0.50
            else "LOW"
        )

        state = (
            f"{dist_cat}_"
            f"S1_{ts.s1_ready}_"
            f"{hp_cat}"
        )

        self.last_farm_state = state

        chosen_action = self._choose_epsilon_greedy(
            domain="farm",
            state_key=state,
            actions=list(self.FARM_ACTIONS),
            epsilon=0.10,
        )

        self.last_farm_action = chosen_action
        return chosen_action

    # ----------------------------------------------------------------------
    # Roam decision
    # ----------------------------------------------------------------------
    def _decide_roam(self, ts: TacticalState) -> str:
        time_since_enemy = max(
            0.0,
            ts.timestamp - self.last_enemy_seen_time,
        )

        if time_since_enemy < 5.0:
            state = "SCOUT_HOT_ZONE"
        elif time_since_enemy < 15.0:
            state = "SCOUT_MID_SEARCH"
        else:
            state = "SCOUT_DEEP_PATROL"

        self.last_roam_state = state

        actions = list(self.ROAM_ACTIONS.keys())

        chosen_action = self._choose_epsilon_greedy(
            domain="roam",
            state_key=state,
            actions=actions,
            epsilon=0.20,
        )

        self.last_roam_action = chosen_action
        return chosen_action

    # ----------------------------------------------------------------------
    # Mapping decision -> typed Action
    # ----------------------------------------------------------------------
    def map_decision_to_action(
        self,
        decision: str,
        ts: TacticalState,
    ) -> Action:
        """Translate string decision into typed Action."""

        action_type = self.DECISION_TO_ACTION_TYPE.get(
            decision,
            ActionType.IDLE,
        )

        player_pos = ts.player_position

        # ==============================================================
        # Combat
        # ==============================================================

        if decision in (
            "TURRET_RETREAT",
            "TACTICAL_RETREAT",
        ):
            safe_point = Vector2(
                player_pos.x - 180.0,
                player_pos.y + 90.0,
            )

            return Action(
                type=action_type,
                direction=safe_point,
                target_id=ts.nearest_enemy_id,
            )

        if decision == "KITE_AND_POKE":
            return Action(
                type=action_type,
                direction=ts.nearest_enemy_position,
                target_id=ts.nearest_enemy_id,
            )

        if decision == "SWEET_SPOT_BURST":
            enemy_pos = (
                ts.nearest_enemy_position
                or player_pos
            )

            dx = enemy_pos.x - player_pos.x
            dy = enemy_pos.y - player_pos.y

            strafe_pos = Vector2(
                player_pos.x - dy * 0.45,
                player_pos.y + dx * 0.45,
            )

            return Action(
                type=action_type,
                direction=strafe_pos,
                target_id=ts.nearest_enemy_id,
            )

        if decision == "DIVE_ALL_IN":
            return Action(
                type=action_type,
                direction=ts.nearest_enemy_position,
                target_id=ts.nearest_enemy_id,
            )

        # ==============================================================
        # Farming
        # ==============================================================

        if decision in (
            "FARM_APPROACH",
            "FARM_KITE_BACK",
        ):
            return Action(
                type=action_type,
                direction=ts.nearest_minion_position,
                target_id=ts.nearest_minion_id,
            )

        if decision == "FARM_S1_AOE":
            return Action(
                type=action_type,
                target_id=ts.nearest_minion_id,
            )

        if decision == "FARM_SWEET_SPOT":
            minion_pos = (
                ts.nearest_minion_position
                or player_pos
            )

            dx = minion_pos.x - player_pos.x
            dy = minion_pos.y - player_pos.y

            strafe_pos = Vector2(
                player_pos.x - dy * 0.40,
                player_pos.y + dx * 0.40,
            )

            return Action(
                type=action_type,
                direction=strafe_pos,
                target_id=ts.nearest_minion_id,
            )

        # ==============================================================
        # Roaming
        # ==============================================================

        if decision in self.ROAM_OFFSETS:
            dx, dy = self.ROAM_OFFSETS[decision]

            target_pt = Vector2(
                player_pos.x + dx,
                player_pos.y + dy,
            )

            return Action(
                type=action_type,
                direction=target_pt,
            )

        # Unknown decision -> safe idle
        return Action(type=ActionType.IDLE)

    # ----------------------------------------------------------------------
    # Learning integration
    # ----------------------------------------------------------------------
    def apply_reward(
        self,
        domain: str,
        state: str,
        action: str,
        reward: Reward,
    ) -> float:
        """Apply reward to QLearningCore."""

        if not isinstance(reward, Reward):
            raise TypeError(
                f"apply_reward expects Reward, got: {type(reward)}"
            )

        return self.q_learning.update_q(
            table_name=domain,
            state_key=state,
            action=action,
            reward=reward.value,
        )
