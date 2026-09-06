"""brain.learning_integration — Event to Reward to Learning Integration for Claude Tactical AI V2.

Closes the cognitive learning pipeline:
    GameEvent(s)
        ↓
    RewardCalculator.for_event()
        ↓
    Reward(s)
        ↓
    LearningLoop.learn_step() (using TransitionBuilder)
        ↓
    QLearningCore.update_td()

Pure cognitive integration:
- Zero duplicated reward constants (delegates strictly to RewardCalculator)
- Zero duplicated Q-learning math (delegates strictly to LearningLoop)
- Zero manual state encoding (delegates strictly to TransitionBuilder / LearningLoop)
- Zero hardware / device / ADB dependencies
"""

from __future__ import annotations

from typing import List, Optional, Union

from actions.models import Action
from brain.learning_loop import LearningLoop
from brain.rewards import Reward, RewardCalculator
from brain.state import TacticalState
from brain.transition import Transition
from brain.transition_builder import TransitionBuilder
from world.events import EventType, GameEvent


class LearningIntegrator:
    """Connects observed GameEvents and tactical transitions to reinforcement learning updates."""

    def __init__(
        self,
        reward_calculator: RewardCalculator,
        transition_builder: TransitionBuilder,
        learning_loop: LearningLoop,
    ) -> None:
        """Initializes LearningIntegrator with injected components.

        Args:
            reward_calculator: RewardCalculator instance for event reward evaluation.
            transition_builder: TransitionBuilder instance for transition construction.
            learning_loop: LearningLoop instance for executing TD updates.
        """
        if reward_calculator is None:
            raise TypeError("reward_calculator cannot be None")
        if transition_builder is None:
            raise TypeError("transition_builder cannot be None")
        if learning_loop is None:
            raise TypeError("learning_loop cannot be None")

        self.reward_calculator = reward_calculator
        self.transition_builder = transition_builder
        self.learning_loop = learning_loop

    def process(
        self,
        state: TacticalState,
        action: Union[Action, str],
        events: List[GameEvent],
        next_state: Optional[TacticalState],
        done: bool,
    ) -> List[Transition]:
        """Processes GameEvents, evaluates Rewards, and executes RL learning updates.

        Pipeline:
            GameEvents
                ↓
            RewardCalculator.for_event()
                ↓
            Reward(s)
                ↓
            LearningLoop.learn_step()
                ↓
            Transition(s)

        Args:
            state: Current TacticalState observation.
            action: Executed action (Action instance or string identifier).
            events: List of GameEvent observations from tracker/vision.
            next_state: Resulting TacticalState observation, or None if done=True.
            done: Terminal flag.

        Returns:
            List[Transition]: Formal Transition tuples created and updated in this step.
        """
        if not events:
            return []

        transitions: List[Transition] = []

        for event in events:
            rewards: List[Reward] = self.reward_calculator.for_event(event)
            if not rewards:
                continue

            # Terminal semantics:
            # If done is explicitly True, or if PLAYER_DEATH indicates a terminal transition
            is_terminal = done or (event.type == EventType.PLAYER_DEATH and next_state is None)
            step_next = None if is_terminal else next_state

            for reward in rewards:
                transition = self.learning_loop.learn_step(
                    state=state,
                    action=action,
                    reward=reward.value,
                    next_state=step_next,
                    done=is_terminal,
                )
                transitions.append(transition)

        return transitions
