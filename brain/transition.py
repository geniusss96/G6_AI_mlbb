"""brain.transition — Reinforcement Learning Transition Model for Claude Tactical AI V2.

Represents an immutable experience tuple (state, action, reward, next_state, done)
for standard Q-learning and experience replay formulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Transition:
    """An immutable specification of an RL transition (s, a, r, s', done).

    Attributes:
        state_key: Abstract discrete identifier of the source state s.
        action: Abstract identifier of the executed action a.
        reward: Scalar reinforcement signal r received after action a.
        next_state_key: Abstract discrete identifier of the resulting state s',
            or None if terminal / unobserved.
        done: Boolean flag indicating if the transition terminates an episode / trajectory.
    """

    state_key: str
    action: str
    reward: float
    next_state_key: Optional[str]
    done: bool

    def __post_init__(self) -> None:
        """Validates structural integrity of the transition tuple."""
        # Validate state_key
        if not isinstance(self.state_key, str):
            raise TypeError(
                f"state_key must be a str, got: {type(self.state_key).__name__}"
            )
        if not self.state_key.strip():
            raise ValueError("state_key cannot be empty")

        # Validate action
        if not isinstance(self.action, str):
            raise TypeError(
                f"action must be a str, got: {type(self.action).__name__}"
            )
        if not self.action.strip():
            raise ValueError("action cannot be empty")

        # Validate reward
        if not isinstance(self.reward, (int, float)):
            raise TypeError(
                f"reward must be a numeric float/int, got: {type(self.reward).__name__}"
            )
        if math.isnan(self.reward) or math.isinf(self.reward):
            raise ValueError(f"reward must be a finite number, got: {self.reward}")

        # Validate next_state_key
        if self.next_state_key is not None:
            if not isinstance(self.next_state_key, str):
                raise TypeError(
                    f"next_state_key must be None or a str, got: {type(self.next_state_key).__name__}"
                )

        # Validate done
        if not isinstance(self.done, bool):
            raise TypeError(f"done must be a bool, got: {type(self.done).__name__}")
