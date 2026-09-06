"""Deterministic simulation runner for Temporal-Difference Q-learning.

Executes synthetic offline RL scenarios and transition sequences in memory,
tracking value updates, convergence, and parameter sensitivities without side effects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from brain.q_learning import QLearningCore
from brain.transition import Transition
from tests.rl_simulation.scenarios import SimulationScenario, SyntheticStep


class SimulationRunner:
    """Controls offline RL training episodes and tracks Q-table state."""

    def __init__(
        self,
        q_core: Optional[QLearningCore] = None,
        learning_rate: float = 0.10,
        discount_factor: float = 0.85,
    ) -> None:
        """Initializes runner with an isolated in-memory QLearningCore."""
        if q_core is not None:
            self.q_core = q_core
        else:
            self.q_core = QLearningCore(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )

    @property
    def alpha(self) -> float:
        return self.q_core.alpha

    @property
    def gamma(self) -> float:
        return self.q_core.gamma

    def get_q(self, state: str, action: str, domain: str = "combat") -> float:
        """Reads Q(state, action) from the specified domain table."""
        return self.q_core.get_q(domain, state, action, default=0.0)

    def set_q(self, state: str, action: str, value: float, domain: str = "combat") -> None:
        """Sets Q(state, action) directly in the specified domain table."""
        self.q_core.set_q(domain, state, action, value)

    def step_transition(
        self,
        transition: Transition,
        table_name: Optional[str] = None,
        alpha: Optional[float] = None,
        next_actions: Optional[List[str]] = None,
    ) -> float:
        """Executes a single update_td step on the in-memory QLearningCore."""
        return self.q_core.update_td(
            transition=transition,
            table_name=table_name,
            alpha=alpha,
            next_actions=next_actions,
        )

    def run_step(
        self,
        step: SyntheticStep,
        domain: str = "combat",
        alpha: Optional[float] = None,
    ) -> float:
        """Executes a single SyntheticStep as a Transition."""
        return self.step_transition(
            transition=step.to_transition(),
            table_name=domain,
            alpha=alpha,
        )

    def run_scenario(
        self,
        scenario: SimulationScenario,
        episodes: int = 1,
        reverse_sweep: bool = False,
        alpha: Optional[float] = None,
    ) -> List[float]:
        """Runs the given scenario for N episodes.

        Args:
            scenario: The simulation scenario containing steps.
            episodes: Number of complete passes over the scenario steps.
            reverse_sweep: If True, steps are processed in reverse order (e.g. terminal backwards)
                           allowing immediate credit assignment to preceding states.
            alpha: Optional custom learning rate.

        Returns:
            List[float]: History of updated Q-values for each step across all episodes.
        """
        history: List[float] = []
        steps = list(scenario.steps)
        if reverse_sweep:
            steps = list(reversed(steps))

        for _ in range(episodes):
            for step in steps:
                val = self.run_step(step, domain=scenario.domain, alpha=alpha)
                history.append(val)

        return history

    def run_repeated_transition(
        self,
        transition: Transition,
        iterations: int = 50,
        alpha: Optional[float] = None,
        table_name: Optional[str] = None,
    ) -> List[float]:
        """Repeatedly updates the exact same transition to track convergence.

        Returns:
            List[float]: Sequence of Q-values after each iteration.
        """
        history: List[float] = []
        for _ in range(iterations):
            val = self.step_transition(
                transition=transition,
                table_name=table_name,
                alpha=alpha,
            )
            history.append(val)
        return history
