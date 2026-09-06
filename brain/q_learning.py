"""brain.q_learning — Centralized Q-Learning mathematical core for Claude Tactical AI V2.

Clean architecture separation:
├── Read-only API (queries without mutating Q-tables)
├── Explicit state creation (deliberate initialization of state vectors)
└── Explicit updates (Bellman moving-average updates and value overrides)

Preserves exact V1 mathematical properties:
- Exponential moving average update: Q(s, a) <- round(Q(s, a) + alpha * [R - Q(s, a)], 2)
- 2-decimal rounding: round(new_q, 2)
- Inactive gamma parameter (0.85) preserved for configuration compatibility
- Discrete Q-tables for distinct operational contexts: roam_q, farm_q, combat_q
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple


class QLearningCore:
    """Isolated Q-Learning core managing state-action tables and updates."""

    def __init__(
        self,
        learning_rate: float = 0.25,
        discount_factor: float = 0.85,
        exploration_rate: float = 0.20,
        roam_q: Optional[Dict[str, Dict[str, float]]] = None,
        farm_q: Optional[Dict[str, Dict[str, float]]] = None,
        combat_q: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.alpha: float = float(learning_rate)
        # Gamma preserved from V1 baseline (inactive in V1 1-step bandit formulation)
        self.gamma: float = float(discount_factor)
        self.epsilon: float = float(exploration_rate)

        # Domain exploration rates matching V1 hardcoded heuristics
        self.roam_epsilon: float = self.epsilon
        self.farm_epsilon: float = 0.10
        self.combat_epsilon: float = 0.08

        # Discrete Q-tables
        self.roam_q_table: Dict[str, Dict[str, float]] = roam_q if roam_q is not None else {}
        self.farm_q_table: Dict[str, Dict[str, float]] = farm_q if farm_q is not None else {}
        self.combat_q_table: Dict[str, Dict[str, float]] = combat_q if combat_q is not None else {}

    def get_table(self, table_name: str) -> Dict[str, Dict[str, float]]:
        """Returns reference to the specified discrete Q-table."""
        name = table_name.lower().strip()
        if name in ("roam", "roam_q", "roam_q_table"):
            return self.roam_q_table
        if name in ("farm", "farm_q", "farm_q_table"):
            return self.farm_q_table
        if name in ("combat", "combat_q", "combat_q_table"):
            return self.combat_q_table
        raise KeyError(f"Unknown Q-table: '{table_name}'. Expected 'roam', 'farm', or 'combat'.")

    # ==========================================================================
    # 1. READ-ONLY API (Zero table mutations)
    # ==========================================================================

    def has_state(self, table_name: str, state_key: str) -> bool:
        """Checks if state exists in the specified table without creating it."""
        table = self.get_table(table_name)
        return state_key in table

    def get_state_actions(
        self, table_name: str, state_key: str
    ) -> Dict[str, float]:
        """Returns a copy of the state's action values or empty dict if not present.

        Read-only: does not insert state_key into table.
        """
        table = self.get_table(table_name)
        actions = table.get(state_key)
        return copy.copy(actions) if actions is not None else {}

    def get_q(
        self,
        table_name: str,
        state_key: str,
        action: str,
        default: float = 0.0,
    ) -> float:
        """Retrieves Q(state, action) value or default if not present.

        Read-only: does not insert state_key into table.
        """
        table = self.get_table(table_name)
        state_dict = table.get(state_key)
        if state_dict is None:
            return default
        return state_dict.get(action, default)

    def get_best_action(
        self,
        table_name: str,
        state_key: str,
        candidate_actions: Optional[List[str]] = None,
        rng: Optional[Any] = None,
    ) -> Tuple[str, float]:
        """Finds action with maximum Q-value for given state.

        Read-only: does not insert state_key into table.
        Preserves V1 tie-breaking (random selection among identical max Q-values).
        """
        table = self.get_table(table_name)
        state_dict = table.get(state_key, {})

        if candidate_actions is not None:
            # Filter by candidates, defaulting to 0.0 for unseen actions
            q_vals = {act: state_dict.get(act, 0.0) for act in candidate_actions}
        else:
            q_vals = state_dict

        if not q_vals:
            raise ValueError(
                f"No actions available in table '{table_name}' for state '{state_key}'"
            )

        max_q = max(q_vals.values())
        best_acts = [act for act, q in q_vals.items() if q == max_q]

        chooser = rng.choice if rng is not None else random.choice
        chosen = chooser(best_acts)
        return chosen, max_q

    def get_table_snapshot(self, table_name: str) -> Dict[str, Dict[str, float]]:
        """Returns deep copy of the domain table for safe inspection."""
        return copy.deepcopy(self.get_table(table_name))

    def export_tables(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Dumps Q-tables for serialization into q_brain.json."""
        return {
            "roam_q": self.roam_q_table,
            "farm_q": self.farm_q_table,
            "combat_q": self.combat_q_table,
        }

    # ==========================================================================
    # 2. EXPLICIT STATE CREATION
    # ==========================================================================

    def init_state(
        self,
        table_name: str,
        state_key: str,
        default_actions: Dict[str, float],
    ) -> bool:
        """Explicitly initializes a state with default action weights if not present.

        Returns True if newly created, False if state already existed.
        """
        table = self.get_table(table_name)
        if state_key in table:
            return False
        table[state_key] = {k: round(float(v), 2) for k, v in default_actions.items()}
        return True

    def ensure_state(
        self,
        table_name: str,
        state_key: str,
        defaults: Dict[str, float],
    ) -> Dict[str, float]:
        """Preserves V1 mutation semantics: initializes state if not present and returns dict."""
        table = self.get_table(table_name)
        if state_key not in table:
            table[state_key] = {k: round(float(v), 2) for k, v in defaults.items()}
        return table[state_key]

    # ==========================================================================
    # 3. EXPLICIT UPDATES
    # ==========================================================================

    def set_q(
        self,
        table_name: str,
        state_key: str,
        action: str,
        value: float,
    ) -> None:
        """Explicitly sets Q(state, action) value rounded to 2 decimal places."""
        table = self.get_table(table_name)
        if state_key not in table:
            table[state_key] = {}
        table[state_key][action] = round(float(value), 2)

    def update_q(
        self,
        table_name: str,
        state_key: str,
        action: str,
        reward: float,
        alpha: Optional[float] = None,
    ) -> float:
        """Applies exact V1 Bellman moving-average update:

            new_q = round(old_q + alpha * (reward - old_q), 2)
        """
        table = self.get_table(table_name)
        if state_key not in table:
            table[state_key] = {action: 0.0}

        effective_alpha = self.alpha if alpha is None else float(alpha)
        old_q = table[state_key].get(action, 0.0)
        new_q = round(old_q + effective_alpha * (reward - old_q), 2)
        table[state_key][action] = new_q
        return new_q

    def reset_state(self, table_name: str, state_key: str) -> bool:
        """Explicitly removes a state from the table. Returns True if removed."""
        table = self.get_table(table_name)
        if state_key in table:
            del table[state_key]
            return True
        return False

    def load_tables(self, data: Dict[str, Any]) -> None:
        """Loads Q-tables from parsed q_brain.json data."""
        self.roam_q_table = data.get("roam_q", {})
        self.farm_q_table = data.get("farm_q", {})
        self.combat_q_table = data.get("combat_q", {})
