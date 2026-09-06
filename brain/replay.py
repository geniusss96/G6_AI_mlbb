"""Experience Replay engine for Claude Tactical AI V2.

Decouples highlight / experience selection and replay ordering from Q-table mutations.
Delegates all mathematical updates strictly to QLearningCore without duplicating
learning equations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from brain.memory import ActionRecord, Experience
from brain.q_learning import QLearningCore


class ExperienceReplay:
    """Experience replay engine executing retroactive reinforcement updates.

    In V1 baseline:
    When a triumph occurs (kill, turret take, burst combo), the sequence of recent
    actions leading up to the moment (up to 6 actions in the last 8 seconds)
    receives an elevated Q-learning update with alpha=0.35.

    ExperienceReplay:
    - Selects transitions and determines replay ordering.
    - Routes domain, state, action, and reward to QLearningCore.
    - Zero local Q-tables, zero local mathematical formulas, zero disk I/O.
    """

    DEFAULT_REPLAY_ALPHA: float = 0.35

    def __init__(self, replay_alpha: float = DEFAULT_REPLAY_ALPHA) -> None:
        self.replay_alpha: float = float(replay_alpha)

    def replay_moment(
        self,
        recent_actions: Sequence[Union[ActionRecord, Dict[str, Any]]],
        reward: float,
        q_learning: QLearningCore,
        alpha: Optional[float] = None,
        only_existing: bool = True,
    ) -> int:
        """Replays a chain of recent actions leading to a triumph moment (V1 semantics).

        Args:
            recent_actions: Sequence of recent actions (e.g. last 6 actions over 8s).
            reward: Triumph reward value to reinforce the action chain with.
            q_learning: Target QLearningCore instance to apply updates to.
            alpha: Replay learning rate (defaults to 0.35 as in V1 baseline).
            only_existing: If True (V1 default), only updates states and actions
                that already exist in the Q-table.

        Returns:
            int: Total number of Q-table updates successfully executed.
        """
        if not isinstance(q_learning, QLearningCore):
            raise TypeError(f"q_learning must be QLearningCore, got {type(q_learning)}")

        eff_alpha = self.replay_alpha if alpha is None else float(alpha)
        updates_count = 0

        for item in recent_actions:
            if isinstance(item, ActionRecord):
                cat = item.category
                state = item.state
                action = item.action
            elif isinstance(item, dict):
                cat = str(item.get("category", ""))
                state = str(item.get("state", ""))
                action = str(item.get("action", ""))
            else:
                continue

            domain = cat.lower().strip()
            if domain not in ("combat", "farm", "roam"):
                continue

            # V1 guard: only updates if table, state, and action already exist
            if only_existing:
                table = q_learning.get_table(domain)
                if state not in table or action not in table[state]:
                    continue

            q_learning.update_q(
                table_name=domain,
                state_key=state,
                action=action,
                reward=reward,
                alpha=eff_alpha,
            )
            updates_count += 1

        return updates_count

    def replay(
        self,
        experiences: Sequence[Any],
        q_learning: QLearningCore,
        default_reward: float = 0.0,
        alpha: Optional[float] = None,
        only_existing: bool = False,
    ) -> int:
        """Executes replay across an arbitrary batch of experience records.

        Accepts:
        - List of (ActionRecord, reward) tuples
        - List of ActionRecord with default_reward
        - List of dicts with keys (category, state, action, [reward])

        Returns:
            int: Number of updates applied.
        """
        if not isinstance(q_learning, QLearningCore):
            raise TypeError(f"q_learning must be QLearningCore, got {type(q_learning)}")

        eff_alpha = self.replay_alpha if alpha is None else float(alpha)
        updates_count = 0

        for exp in experiences:
            reward = default_reward

            if isinstance(exp, tuple) and len(exp) >= 2:
                record, item_reward = exp[0], float(exp[1])
                reward = item_reward
            else:
                record = exp

            if isinstance(record, ActionRecord):
                cat = record.category
                state = record.state
                action = record.action
            elif isinstance(record, dict):
                cat = str(record.get("category", ""))
                state = str(record.get("state", ""))
                action = str(record.get("action", ""))
                if "reward" in record:
                    reward = float(record["reward"])
            else:
                continue

            domain = cat.lower().strip()
            if domain not in ("combat", "farm", "roam"):
                continue

            if only_existing:
                table = q_learning.get_table(domain)
                if state not in table or action not in table[state]:
                    continue

            q_learning.update_q(
                table_name=domain,
                state_key=state,
                action=action,
                reward=reward,
                alpha=eff_alpha,
            )
            updates_count += 1

        return updates_count
