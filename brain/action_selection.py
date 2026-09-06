"""brain.action_selection — Action Selection & Exploration Layer for Claude Tactical AI V2.

Selects an abstract Action based on:
- TacticalState observation
- Discrete state encoding via TacticalStateEncoder
- Q-values from QLearningCore (get_best_action)
- Epsilon-greedy exploration policy via ExplorationPolicy

Pure decision layer:
- Zero Q-value updates (no update_q or update_td)
- Zero memory writes
- Zero reward calculations
- Zero hardware / device / ADB dependencies
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from actions.models import Action, ActionType
from brain.encoder import TacticalStateEncoder
from brain.exploration import ExplorationContext, ExplorationPolicy
from brain.q_learning import QLearningCore
from brain.state import TacticalState
from world.models import Vector2


class ActionSelector:
    """Selects battlefield Action using epsilon-greedy exploration and Q-values."""

    # Canonical 1:1 mapping from V1 decision strings to V2 typed ActionType
    DECISION_TO_ACTION_TYPE: Dict[str, ActionType] = {
        # V1 Combat Decisions
        "TURRET_RETREAT": ActionType.RETREAT,
        "TACTICAL_RETREAT": ActionType.RETREAT,
        "KITE_AND_POKE": ActionType.KITE,
        "SWEET_SPOT_BURST": ActionType.ATTACK,
        "DIVE_ALL_IN": ActionType.CAST_ULT,
        # V1 Farm Decisions
        "FARM_APPROACH": ActionType.MOVE,
        "FARM_KITE_BACK": ActionType.KITE,
        "FARM_SWEET_SPOT": ActionType.FARM,
        "FARM_S1_AOE": ActionType.CAST_S1,
        # V1 Roam Decisions
        "LANE_ADVANCE": ActionType.MOVE,
        "LANE_PUSH": ActionType.MOVE,
        "RIVER_SCOUT": ActionType.SCOUT,
        "FLANK_ADVANCE": ActionType.SCOUT,
        "CREEP_INTERCEPT": ActionType.FARM,
        # Direct ActionType string mappings
        "MOVE": ActionType.MOVE,
        "ATTACK": ActionType.ATTACK,
        "CAST_S1": ActionType.CAST_S1,
        "CAST_S2": ActionType.CAST_S2,
        "CAST_ULT": ActionType.CAST_ULT,
        "RETREAT": ActionType.RETREAT,
        "FARM": ActionType.FARM,
        "KITE": ActionType.KITE,
        "SCOUT": ActionType.SCOUT,
        "IDLE": ActionType.IDLE,
        "move": ActionType.MOVE,
        "attack": ActionType.ATTACK,
        "cast_s1": ActionType.CAST_S1,
        "cast_s2": ActionType.CAST_S2,
        "cast_ult": ActionType.CAST_ULT,
        "retreat": ActionType.RETREAT,
        "farm": ActionType.FARM,
        "kite": ActionType.KITE,
        "scout": ActionType.SCOUT,
        "idle": ActionType.IDLE,
    }

    # Roam target direction offsets (from V1 baseline)
    ROAM_OFFSETS: Dict[str, Tuple[float, float]] = {
        "LANE_ADVANCE": (120.0, 0.0),
        "LANE_PUSH": (180.0, 0.0),
        "RIVER_SCOUT": (0.0, -150.0),
        "FLANK_ADVANCE": (80.0, 80.0),
    }

    def __init__(
        self,
        encoder: TacticalStateEncoder,
        q_learning: QLearningCore,
        exploration: ExplorationPolicy,
    ) -> None:
        """Initializes ActionSelector with injected components.

        Args:
            encoder: TacticalStateEncoder for state key discretization.
            q_learning: QLearningCore for reading Q-values.
            exploration: ExplorationPolicy for evaluating exploration rate.
        """
        if encoder is None:
            raise TypeError("encoder cannot be None")
        if q_learning is None:
            raise TypeError("q_learning cannot be None")
        if exploration is None:
            raise TypeError("exploration cannot be None")

        self.encoder = encoder
        self.q_learning = q_learning
        self.exploration = exploration

    def _normalize_context(
        self, context: Union[ExplorationContext, str]
    ) -> ExplorationContext:
        """Validates and normalizes context to ExplorationContext enum."""
        if isinstance(context, ExplorationContext):
            return context
        if isinstance(context, str):
            ctx_clean = context.lower().strip()
            for member in ExplorationContext:
                if member.value == ctx_clean:
                    return member
        raise ValueError(
            f"Unknown or invalid exploration context: {context}. "
            f"Allowed contexts are: {[c.value for c in ExplorationContext]}"
        )

    def to_action_type(self, decision: str) -> ActionType:
        """Maps decision string identifier to typed ActionType enum."""
        return self.DECISION_TO_ACTION_TYPE.get(decision, ActionType.IDLE)

    def to_action(self, decision: str, ts: TacticalState) -> Action:
        """Translates string decision into typed Action with target coordinates."""
        action_type = self.to_action_type(decision)
        player_pos = ts.player_position

        # --- Combat Decisions ---
        if decision in ("TURRET_RETREAT", "TACTICAL_RETREAT"):
            safe_point = Vector2(player_pos.x - 180.0, player_pos.y + 90.0)
            return Action(type=action_type, direction=safe_point, target_id=ts.nearest_enemy_id)

        if decision in ("KITE_AND_POKE", "DIVE_ALL_IN") or action_type in (ActionType.KITE, ActionType.CAST_ULT):
            return Action(type=action_type, direction=ts.nearest_enemy_position, target_id=ts.nearest_enemy_id)

        if decision == "SWEET_SPOT_BURST":
            enemy_pos = ts.nearest_enemy_position or player_pos
            dx = enemy_pos.x - player_pos.x
            dy = enemy_pos.y - player_pos.y
            strafe_pos = Vector2(player_pos.x - dy * 0.45, player_pos.y + dx * 0.45)
            return Action(type=action_type, direction=strafe_pos, target_id=ts.nearest_enemy_id)

        # --- Farm Decisions ---
        if decision in ("FARM_APPROACH", "FARM_KITE_BACK"):
            return Action(type=action_type, direction=ts.nearest_minion_position, target_id=ts.nearest_minion_id)

        if decision == "FARM_S1_AOE":
            return Action(type=action_type, target_id=ts.nearest_minion_id)

        if decision == "FARM_SWEET_SPOT":
            minion_pos = ts.nearest_minion_position or player_pos
            dx = minion_pos.x - player_pos.x
            dy = minion_pos.y - player_pos.y
            strafe_pos = Vector2(player_pos.x - dy * 0.40, player_pos.y + dx * 0.40)
            return Action(type=action_type, direction=strafe_pos, target_id=ts.nearest_minion_id)

        # --- Roam Decisions ---
        if decision in self.ROAM_OFFSETS:
            dx, dy = self.ROAM_OFFSETS[decision]
            target_pt = Vector2(player_pos.x + dx, player_pos.y + dy)
            return Action(type=action_type, direction=target_pt)

        # Target preservation fallback
        if ts.nearest_enemy_position is not None:
            return Action(type=action_type, direction=ts.nearest_enemy_position, target_id=ts.nearest_enemy_id)
        if ts.nearest_minion_position is not None:
            return Action(type=action_type, direction=ts.nearest_minion_position, target_id=ts.nearest_minion_id)

        return Action(type=action_type)

    def select_decision(
        self,
        state: TacticalState,
        context: Union[ExplorationContext, str],
        candidate_actions: List[str],
        rng: Optional[Any] = None,
    ) -> str:
        """Selects the winning decision string using epsilon-greedy policy.

        Args:
            state: Observed TacticalState.
            context: ExplorationContext (ROAM, FARM, COMBAT).
            candidate_actions: Non-empty list of available action strings.
            rng: Optional random number generator (defaults to random).

        Returns:
            str: Selected decision string identifier.
        """
        if not candidate_actions:
            raise ValueError("candidate_actions cannot be empty")

        ctx = self._normalize_context(context)

        # 1. Evaluate exploration decision (random choice)
        if self.exploration.should_explore(ctx, rng=rng):
            chooser = rng.choice if rng is not None else random.choice
            return chooser(candidate_actions)

        # 2. Exploitation: argmax_a Q(s, a)
        state_key = self.encoder.encode(state, context=ctx.value)
        chosen, _ = self.q_learning.get_best_action(
            table_name=ctx.value,
            state_key=state_key,
            candidate_actions=candidate_actions,
            rng=rng,
        )
        return chosen

    def select(
        self,
        state: TacticalState,
        context: Union[ExplorationContext, str],
        candidate_actions: List[str],
        rng: Optional[Any] = None,
    ) -> Action:
        """Selects an Action using epsilon-greedy policy.

        Args:
            state: Observed TacticalState.
            context: Operational context (ROAM, FARM, COMBAT).
            candidate_actions: Non-empty list of allowed candidate action strings.
            rng: Optional deterministic RNG.

        Returns:
            Action: Complete typed Action with coordinates.
        """
        decision = self.select_decision(
            state=state,
            context=context,
            candidate_actions=candidate_actions,
            rng=rng,
        )
        return self.to_action(decision, state)
