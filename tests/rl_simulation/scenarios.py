"""Synthetic deterministic RL environment and test scenarios.

Models discrete (state, action, reward, next_state, done) transitions
for offline evaluation of Temporal-Difference Q-learning without game or CV/ADB dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from brain.transition import Transition


@dataclass(frozen=True)
class SyntheticStep:
    """An immutable atomic step in the synthetic RL environment."""

    state: str
    action: str
    reward: float
    next_state: Optional[str]
    done: bool

    def to_transition(self) -> Transition:
        """Converts step to a formal V2 Transition dataclass."""
        return Transition(
            state_key=self.state,
            action=self.action,
            reward=float(self.reward),
            next_state_key=self.next_state,
            done=self.done,
        )


@dataclass(frozen=True)
class SimulationScenario:
    """An immutable scenario composed of ordered synthetic steps."""

    name: str
    steps: Tuple[SyntheticStep, ...]
    domain: str = "combat"
    description: str = ""


class SyntheticEnvironment:
    """Minimal deterministic synthetic environment.

    Allows registering transitions (state, action) -> (reward, next_state, done)
    and stepping through them deterministically.
    """

    def __init__(self, initial_state: str = "S0") -> None:
        self.initial_state = initial_state
        self.current_state: Optional[str] = initial_state
        self._transitions: Dict[Tuple[str, str], Tuple[float, Optional[str], bool]] = {}

    def add_transition(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str],
        done: bool,
    ) -> None:
        """Registers a deterministic transition rule."""
        if not done and next_state is None:
            raise ValueError("next_state cannot be None for non-terminal transition")
        self._transitions[(state, action)] = (float(reward), next_state, done)

    def reset(self, state: Optional[str] = None) -> str:
        """Resets the environment to initial or specified state."""
        self.current_state = state or self.initial_state
        return self.current_state

    def step(self, action: str) -> SyntheticStep:
        """Executes action in current state and transitions environment."""
        if self.current_state is None:
            raise RuntimeError("Cannot step in a terminated environment. Call reset() first.")

        key = (self.current_state, action)
        if key not in self._transitions:
            raise KeyError(f"No transition registered for state '{self.current_state}' and action '{action}'")

        reward, next_state, done = self._transitions[key]
        step = SyntheticStep(
            state=self.current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.current_state = None if done else next_state
        return step


# ---------------------------------------------------------------------------
# Canonical Predefined Scenarios for Offline RL Simulation
# ---------------------------------------------------------------------------

def create_positive_future_scenario() -> SimulationScenario:
    """Scenario 1: Positive future reward propagation.

    Path: S0 --KITE--> S1 --IDLE--> terminal
    Rewards: KITE: 0.0, IDLE: +10.0
    """
    steps = (
        SyntheticStep(state="S0", action="KITE", reward=0.0, next_state="S1", done=False),
        SyntheticStep(state="S1", action="IDLE", reward=10.0, next_state=None, done=True),
    )
    return SimulationScenario(
        name="positive_future_value",
        steps=steps,
        domain="combat",
        description="S0 --KITE--> S1 --IDLE--> terminal with positive terminal reward",
    )


def create_negative_future_scenario() -> SimulationScenario:
    """Scenario 2: Negative future reward propagation.

    Path: S0 --KITE--> S1 --BAD_ACTION--> terminal
    Rewards: KITE: 0.0, BAD_ACTION: -20.0
    """
    steps = (
        SyntheticStep(state="S0", action="KITE", reward=0.0, next_state="S1", done=False),
        SyntheticStep(state="S1", action="BAD_ACTION", reward=-20.0, next_state=None, done=True),
    )
    return SimulationScenario(
        name="negative_future_value",
        steps=steps,
        domain="combat",
        description="S0 --KITE--> S1 --BAD_ACTION--> terminal with penalty",
    )


def create_immediate_vs_future_scenarios() -> Tuple[SimulationScenario, SimulationScenario]:
    """Scenario 3: Immediate reward vs future reward trade-off.

    Action A: S0 --ACTION_A--> terminal (+5.0 immediate)
    Action B: S0 --ACTION_B--> S_B --IDLE--> terminal (0.0 immediate, +10.0 future)
    """
    scenario_a = SimulationScenario(
        name="immediate_reward_a",
        steps=(
            SyntheticStep(state="S0", action="ACTION_A", reward=5.0, next_state=None, done=True),
        ),
        domain="combat",
        description="Immediate +5 reward at terminal",
    )
    scenario_b = SimulationScenario(
        name="delayed_reward_b",
        steps=(
            SyntheticStep(state="S0", action="ACTION_B", reward=0.0, next_state="S_B", done=False),
            SyntheticStep(state="S_B", action="IDLE", reward=10.0, next_state=None, done=True),
        ),
        domain="combat",
        description="Delayed +10 reward at S_B",
    )
    return scenario_a, scenario_b


def create_terminal_scenario() -> SimulationScenario:
    """Scenario: Terminal transition without next state."""
    steps = (
        SyntheticStep(state="S0", action="FINISH", reward=15.0, next_state=None, done=True),
    )
    return SimulationScenario(
        name="terminal_transition",
        steps=steps,
        domain="combat",
        description="Single terminal step (done=True, next_state=None)",
    )


def create_greedy_choice_scenario() -> Tuple[SimulationScenario, List[SyntheticStep]]:
    """Scenario: S0 -> S1 where S1 has multiple candidate actions with different values.

    Preparation:
      S1 has ACTION_LOW (Q = 5.0) and ACTION_HIGH (Q = 25.0)
    Step evaluated:
      S0 --MOVE--> S1 (reward = 2.0)
    """
    prep_steps = [
        SyntheticStep(state="S1", action="ACTION_LOW", reward=5.0, next_state=None, done=True),
        SyntheticStep(state="S1", action="ACTION_HIGH", reward=25.0, next_state=None, done=True),
    ]
    transition_scenario = SimulationScenario(
        name="greedy_choice",
        steps=(
            SyntheticStep(state="S0", action="MOVE", reward=2.0, next_state="S1", done=False),
        ),
        domain="combat",
        description="S0 leads to S1 where candidate values are 5 and 25",
    )
    return transition_scenario, prep_steps
