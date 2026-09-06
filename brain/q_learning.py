"""
brain.q_learning — Centralized Q-Learning mathematical core for Claude Tactical AI V2.

Extracted 1:1 from V1 ClaudeRLBrain baseline.
Preserves exact V1 mathematical properties:
- Exponential moving average update: Q(s, a) <- Q(s, a) + alpha * [R - Q(s, a)]
- 2-decimal rounding: round(new_q, 2)
- Discount factor gamma (0.85) preserved as configuration but inactive in 1-step bandit updates
- Discrete Q-tables for distinct tactical contexts: roam_q, farm_q, combat_q
- Lazy state initialization and tie-breaking action selection

Pure algorithmic and mathematical module: no hardware, no screen coordinates, no tactical decision trees.
"""

from typing import Dict, List, Optional, Tuple, Any
import random


class QLearningCore:
    """
    Isolated Q-Learning core managing state-action tables and updates.
    """

    def __init__(
        self,
        learning_rate: float = 0.25,
        discount_factor: float = 0.85,
        exploration_rate: float = 0.20,
        roam_q: Optional[Dict[str, Dict[str, float]]] = None,
        farm_q: Optional[Dict[str, Dict[str, float]]] = None,
        combat_q: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.alpha = float(learning_rate)
        # Gamma preserved from V1 baseline (inactive in V1 1-step bandit formulation)
        self.gamma = float(discount_factor)
        self.epsilon = float(exploration_rate)

        # Domain exploration rates matching V1 hardcoded heuristics
        self.roam_epsilon = self.epsilon
        self.farm_epsilon = 0.10
        self.combat_epsilon = 0.08

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

    def ensure_state(
        self,
        table_name: str,
        state_key: str,
        defaults: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Preserves V1 mutation semantics: if state_key is not in the table,
        initializes it with default action values.
        """
        table = self.get_table(table_name)
        if state_key not in table:
            table[state_key] = dict(defaults)
        return table[state_key]

    def get_q(
        self,
        table_name: str,
        state_key: str,
        action: str,
        default: float = 0.0,
    ) -> float:
        """Retrieves Q(state, action) value or default if not present."""
        table = self.get_table(table_name)
        state_dict = table.get(state_key)
        if state_dict is None:
            return default
        return state_dict.get(action, default)

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
        """
        Applies exact V1 Bellman moving-average update:
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

    def get_best_action(
        self,
        table_name: str,
        state_key: str,
        candidate_actions: Optional[List[str]] = None,
        rng: Optional[Any] = None,
    ) -> Tuple[str, float]:
        """
        Finds action with maximum Q-value for given state.
        Preserves V1 tie-breaking (random selection among identical max Q-values).
        """
        table = self.get_table(table_name)
        state_dict = table.get(state_key, {})

        if candidate_actions is not None:
            # Filter by candidates, using 0.0 for unseen actions
            q_vals = {act: state_dict.get(act, 0.0) for act in candidate_actions}
        else:
            q_vals = state_dict

        if not q_vals:
            raise ValueError(f"No actions available in table '{table_name}' for state '{state_key}'")

        max_q = max(q_vals.values())
        best_acts = [act for act, q in q_vals.items() if q == max_q]

        chooser = rng.choice if rng is not None else random.choice
        chosen = chooser(best_acts)
        return chosen, max_q

    def export_tables(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Dumps Q-tables for serialization into q_brain.json."""
        return {
            "roam_q": self.roam_q_table,
            "farm_q": self.farm_q_table,
            "combat_q": self.combat_q_table,
        }

    def load_tables(self, data: Dict[str, Any]) -> None:
        """Loads Q-tables from parsed q_brain.json data."""
        self.roam_q_table = data.get("roam_q", {})
        self.farm_q_table = data.get("farm_q", {})
        self.combat_q_table = data.get("combat_q", {})
