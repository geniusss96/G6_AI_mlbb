"""Exploration policy abstraction for Claude Tactical AI V2.

Extracts the V1 epsilon-greedy exploration probabilities and selection policy
without coupling to Q-tables, rewards, or tactical decision logic.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Any, Dict, Optional, Union


class ExplorationContext(str, Enum):
    """Battlefield operational context for exploration."""

    ROAM = "roam"
    FARM = "farm"
    COMBAT = "combat"


class ExplorationPolicy:
    """Evaluates whether to explore vs exploit based on context.

    Preserves exact V1 baseline exploration probabilities:
    - ROAM: 0.20 (20% random exploration of map pathways)
    - FARM: 0.10 (10% exploration of creep farming stances)
    - COMBAT: 0.08 (8% exploration of duel actions)
    """

    DEFAULT_ROAM_EPSILON: float = 0.20
    DEFAULT_FARM_EPSILON: float = 0.10
    DEFAULT_COMBAT_EPSILON: float = 0.08

    def __init__(
        self,
        roam_epsilon: float = DEFAULT_ROAM_EPSILON,
        farm_epsilon: float = DEFAULT_FARM_EPSILON,
        combat_epsilon: float = DEFAULT_COMBAT_EPSILON,
        default_rng: Optional[Any] = None,
    ) -> None:
        self._rates: Dict[ExplorationContext, float] = {
            ExplorationContext.ROAM: float(roam_epsilon),
            ExplorationContext.FARM: float(farm_epsilon),
            ExplorationContext.COMBAT: float(combat_epsilon),
        }
        self._rng: Any = default_rng if default_rng is not None else random

    def _normalize_context(
        self, context: Union[ExplorationContext, str]
    ) -> ExplorationContext:
        if isinstance(context, ExplorationContext):
            return context
        try:
            return ExplorationContext(str(context).lower().strip())
        except ValueError:
            raise ValueError(
                f"Unknown exploration context: '{context}'. Expected 'roam', 'farm', or 'combat'."
            )

    def epsilon(self, context: Union[ExplorationContext, str]) -> float:
        """Returns the exploration rate (epsilon) for the given context."""
        ctx = self._normalize_context(context)
        return self._rates[ctx]

    def set_epsilon(
        self, context: Union[ExplorationContext, str], value: float
    ) -> None:
        """Sets custom exploration rate for context (0.0 <= value <= 1.0)."""
        ctx = self._normalize_context(context)
        val = float(value)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Epsilon must be in [0.0, 1.0], got {val}")
        self._rates[ctx] = val

    def should_explore(
        self,
        context: Union[ExplorationContext, str],
        rng: Optional[Any] = None,
    ) -> bool:
        """Determines if the agent should explore (random choice) in the given context.

        Returns True if random() < epsilon(context), False otherwise.
        """
        eps = self.epsilon(context)
        active_rng = rng if rng is not None else self._rng
        return active_rng.random() < eps
