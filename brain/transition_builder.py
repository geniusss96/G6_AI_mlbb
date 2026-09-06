"""brain.transition_builder — Transition Builder for Claude Tactical AI V2.

Constructs formal RL Transition instances (s, a, r, s', done) from TacticalStates,
Actions, and numerical rewards using TacticalStateEncoder.

Pure builder:
- Zero Q-learning updates or reads
- Zero reward calculation (accepts precomputed rewards)
- Zero hardware / device / timer side-effects
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from actions.models import Action
from brain.encoder import TacticalStateEncoder
from brain.rewards import Reward
from brain.state import TacticalState
from brain.transition import Transition


class TransitionBuilder:
    """Constructs validated Transition tuples from observed tactical states and actions."""

    def __init__(self, encoder: Optional[TacticalStateEncoder] = None) -> None:
        self.encoder: TacticalStateEncoder = (
            encoder if encoder is not None else TacticalStateEncoder()
        )

    def build(
        self,
        state: TacticalState,
        action: Union[Action, str],
        reward: Union[float, int, Reward],
        next_state: Optional[TacticalState],
        done: bool,
        context: Optional[str] = None,
        next_context: Optional[str] = None,
        last_enemy_seen_time: float = 0.0,
        next_last_enemy_seen_time: float = 0.0,
    ) -> Transition:
        """Constructs an immutable Transition tuple.

        Args:
            state: Current observation snapshot (TacticalState).
            action: Executed intent (Action object or action string identifier).
            reward: Precomputed numeric reinforcement signal or Reward object.
            next_state: Resulting observation snapshot, or None if done=True.
            done: Terminal flag indicating end of episode or trajectory.
            context: Optional domain context override ("combat", "farm", "roam") for state.
            next_context: Optional domain context override for next_state.
            last_enemy_seen_time: Reference timestamp for roam discretization of state.
            next_last_enemy_seen_time: Reference timestamp for roam discretization of next_state.

        Returns:
            Transition: Validated immutable (state_key, action, reward, next_state_key, done) tuple.

        Raises:
            TypeError: If types of state, action, reward, next_state, or done are invalid.
            ValueError: If reward is non-finite or next_state is missing when done is False.
        """
        # 1. Validate state
        if not isinstance(state, TacticalState):
            raise TypeError(
                f"state must be a TacticalState instance, got: {type(state).__name__}"
            )

        # 2. Validate and extract action identifier
        if isinstance(action, Action):
            action_name = action.type.value
        elif isinstance(action, str):
            action_name = action.strip()
        else:
            raise TypeError(
                f"action must be Action or str, got: {type(action).__name__}"
            )

        if not action_name:
            raise ValueError("action identifier cannot be empty")

        # 3. Validate and extract reward
        if isinstance(reward, Reward):
            reward_val = float(reward.value)
        elif isinstance(reward, (int, float)):
            reward_val = float(reward)
        else:
            raise TypeError(
                f"reward must be a numeric float/int or Reward, got: {type(reward).__name__}"
            )

        if math.isnan(reward_val) or math.isinf(reward_val):
            raise ValueError(f"reward must be a finite number, got: {reward_val}")

        # 4. Validate done flag
        if not isinstance(done, bool):
            raise TypeError(f"done must be a bool, got: {type(done).__name__}")

        # 5. Validate terminal / next_state consistency
        if not done and next_state is None:
            raise ValueError("next_state cannot be None when done is False")

        if next_state is not None and not isinstance(next_state, TacticalState):
            raise TypeError(
                f"next_state must be TacticalState or None, got: {type(next_state).__name__}"
            )

        # 6. Encode discrete state keys via TacticalStateEncoder
        state_key = self.encoder.encode(
            state,
            context=context,
            last_enemy_seen_time=last_enemy_seen_time,
        )

        next_state_key: Optional[str] = None
        if next_state is not None:
            next_state_key = self.encoder.encode(
                next_state,
                context=next_context,
                last_enemy_seen_time=next_last_enemy_seen_time,
            )

        return Transition(
            state_key=state_key,
            action=action_name,
            reward=reward_val,
            next_state_key=next_state_key,
            done=done,
        )
