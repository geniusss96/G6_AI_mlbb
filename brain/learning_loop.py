"""brain.learning_loop — Orchestration Layer for Single RL Learning Step.

Coordinates the reinforcement learning step pipeline for Claude Tactical AI V2:
    TacticalState
        ↓
    TacticalStateEncoder
        ↓
    TransitionBuilder
        ↓
    Transition
        ↓
    QLearningCore.update_td()
        ↓
    BrainMemory (optional action logging)

Pure orchestration:
- Zero action selection (Action arrives externally)
- Zero reward calculation (reward arrives precomputed)
- Zero automatic experience replay
- Zero hardware / device / timer side-effects
"""

from __future__ import annotations

from typing import Optional, Union

from actions.models import Action
from brain.encoder import TacticalStateEncoder
from brain.memory import BrainMemory
from brain.q_learning import QLearningCore
from brain.state import TacticalState
from brain.transition import Transition
from brain.transition_builder import TransitionBuilder


class LearningLoop:
    """Orchestrates a single reinforcement learning update step for Claude Tactical AI V2."""

    def __init__(
        self,
        encoder: TacticalStateEncoder,
        transition_builder: TransitionBuilder,
        q_learning: QLearningCore,
        memory: Optional[BrainMemory] = None,
    ) -> None:
        """Initializes the LearningLoop with injected dependencies.

        Args:
            encoder: TacticalStateEncoder instance for state discretization.
            transition_builder: TransitionBuilder instance for building Transition objects.
            q_learning: QLearningCore instance for Q-table updates via update_td.
            memory: Optional BrainMemory instance for logging action history.
        """
        if encoder is None:
            raise TypeError("encoder cannot be None")
        if transition_builder is None:
            raise TypeError("transition_builder cannot be None")
        if q_learning is None:
            raise TypeError("q_learning cannot be None")

        self.encoder = encoder
        self.transition_builder = transition_builder
        # Ensure transition_builder is bound to the injected encoder
        if hasattr(self.transition_builder, "encoder"):
            self.transition_builder.encoder = self.encoder

        self.q_learning = q_learning
        self.memory = memory

    def _infer_category(self, state: TacticalState, state_key: str) -> str:
        """Infers the tactical category ('COMBAT', 'FARM', 'ROAM') for memory logging."""
        if state.enemy_visible and state.nearest_enemy_distance is not None:
            return "COMBAT"
        if state.minion_count > 0 and state.nearest_minion_distance is not None:
            return "FARM"
        if state_key.startswith("SCOUT_"):
            return "ROAM"
        if state_key.startswith("CREEP_"):
            return "FARM"
        return "COMBAT"

    def learn_step(
        self,
        state: TacticalState,
        action: Union[Action, str],
        reward: float,
        next_state: Optional[TacticalState],
        done: bool,
    ) -> Transition:
        """Executes a single TD Q-learning update step.

        Pipeline:
        1. Encodes state and builds immutable Transition via TransitionBuilder.
        2. Executes QLearningCore.update_td(transition).
        3. Logs action to BrainMemory if memory is configured.
        4. Returns the constructed Transition.

        Args:
            state: Current TacticalState observation.
            action: Executed action (Action instance or string identifier).
            reward: Precomputed numeric scalar reward.
            next_state: Resulting TacticalState observation, or None if done=True.
            done: Terminal flag.

        Returns:
            Transition: The formal immutable Transition tuple created for this step.

        Raises:
            TypeError: If arguments are of invalid type.
            ValueError: If reward is non-finite or next_state is None when done=False.
        """
        # 1. Build Transition through TransitionBuilder
        transition = self.transition_builder.build(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        # 2. Execute V2 Temporal-Difference Q-learning update
        self.q_learning.update_td(transition)

        # 3. Optional memory logging (sliding action history)
        if self.memory is not None:
            category = self._infer_category(state, transition.state_key)
            self.memory.log_action(
                category=category,
                state=transition.state_key,
                action=transition.action,
                timestamp=state.timestamp,
            )

        return transition
