"""Regression runner executing comparisons between V1 and V2 brain decisions."""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

from actions.models import ActionType
from brain.brain import ClaudeBrainV2
from brain.memory import BrainMemory
from brain.q_learning import QLearningCore
from brain.state import TacticalState
from claude_brain import ClaudeRLBrain
from tests.brain_regression.scenarios import (
    BrainScenario,
    V1_TO_V2_ACTION_MAP,
    create_scenario_q_tables,
)


@dataclass(frozen=True)
class RegressionResult:
    """Evaluation result comparing V1 legacy decision against V2 ActionType."""

    scenario: str
    v1_decision: str
    v2_action: ActionType
    match: bool
    reason: str


class ExploitRNG:
    """Deterministic random mock to guarantee exploitation in regression runs."""

    def __init__(self, fixed_random: float = 0.99) -> None:
        self.fixed_random = fixed_random

    def random(self) -> float:
        return self.fixed_random

    def choice(self, seq: Any) -> Any:
        return seq[0]


def evaluate_v1_decision(v1_brain: Any, ts: TacticalState) -> str:
    """Queries V1 ClaudeRLBrain with extracted TacticalState parameters."""
    if ts.enemy_visible and ts.nearest_enemy_distance is not None:
        return v1_brain.evaluate_combat_stance(
            dist_enemy=ts.nearest_enemy_distance,
            can_combo=ts.can_combo,
            is_turret_near=ts.enemy_near_turret,
            hp_self_ratio=ts.player_hp_ratio,
        )

    if ts.minion_count > 0 and ts.nearest_minion_distance is not None:
        return v1_brain.evaluate_farm_stance(
            dist_farm=ts.nearest_minion_distance,
            s1_ok=ts.s1_ready,
            hp_self_ratio=ts.player_hp_ratio,
        )

    res = v1_brain.choose_roam_direction(
        ref_pt=ts.player_position.as_tuple(),
        time_since_enemy=ts.timestamp,
    )
    if isinstance(res, tuple) and len(res) == 2:
        return res[0]
    return str(res)


def create_isolated_brains(
    target_q_tables: Optional[Tuple[Any, Any, Any]] = None,
) -> Tuple[ClaudeRLBrain, ClaudeBrainV2]:
    """Creates isolated V1 and V2 brains with identical Q-tables and zero disk writes."""
    roam_q, farm_q, combat_q = target_q_tables or create_scenario_q_tables()

    # V1 Brain with non-existent memory file and disabled disk save
    v1 = ClaudeRLBrain(
        memory_file="__regression_dummy_isolated__.json",
        exploration_rate=0.0,
    )
    v1.save_memory = lambda: None  # Guaranteed no disk write
    v1.roam_q_table = copy.deepcopy(roam_q)
    v1.farm_q_table = copy.deepcopy(farm_q)
    v1.combat_q_table = copy.deepcopy(combat_q)

    # V2 Brain with matching Q-tables and deterministic RNG
    v2_q = QLearningCore(exploration_rate=0.0)
    v2_q.roam_q_table = copy.deepcopy(roam_q)
    v2_q.farm_q_table = copy.deepcopy(farm_q)
    v2_q.combat_q_table = copy.deepcopy(combat_q)

    v2 = ClaudeBrainV2(
        q_learning=v2_q,
        memory=BrainMemory(memory_file="__regression_dummy_isolated__.json"),
        random_generator=ExploitRNG(fixed_random=0.99),
    )
    v2.memory.save = lambda *args, **kwargs: True  # Guaranteed no disk write

    return v1, v2


def run_regression(
    scenarios: List[BrainScenario],
    v1_brain: Optional[Any] = None,
    v2_brain: Optional[Any] = None,
) -> List[RegressionResult]:
    """Runs regression evaluation across the provided scenario list.

    Compares V1 decision against V2 ActionType using the canonical mapping.
    """
    results: List[RegressionResult] = []

    # If brains not provided, create isolated deterministic instances
    active_v1 = v1_brain
    active_v2 = v2_brain
    if active_v1 is None or active_v2 is None:
        default_v1, default_v2 = create_isolated_brains()
        if active_v1 is None:
            active_v1 = default_v1
        if active_v2 is None:
            active_v2 = default_v2

    # Patch random.random for V1 during evaluation to force deterministic exploitation
    with patch("random.random", return_value=0.99):
        for scenario in scenarios:
            # Reset transient history between isolated scenarios
            active_v2.last_enemy_seen_time = 0.0

            # 1. Obtain V1 decision
            # For roam scenarios targeting specific directions, set the peak Q in tables
            if scenario.expected_v1_decision in ("FLANK_ADVANCE", "CREEP_INTERCEPT"):
                active_v1.roam_q_table["SCOUT_DEEP_PATROL"] = {
                    act: (100.0 if act == scenario.expected_v1_decision else 0.0)
                    for act in active_v1.ROAM_ACTIONS.keys()
                }
                active_v2.q_learning.roam_q_table["SCOUT_DEEP_PATROL"] = {
                    act: (100.0 if act == scenario.expected_v1_decision else 0.0)
                    for act in active_v2.ROAM_ACTIONS.keys()
                }

            v1_decision = evaluate_v1_decision(active_v1, scenario.state)

            # 2. Obtain V2 action
            v2_action_obj = active_v2.decide(scenario.state)
            v2_action = v2_action_obj.type

            # 3. Compare semantic mapping
            expected_v2_action = V1_TO_V2_ACTION_MAP.get(v1_decision)

            if expected_v2_action is None:
                match = False
                reason = f"NOT_COMPARABLE: V1 '{v1_decision}' has no V2 mapping"
            elif v2_action == expected_v2_action:
                match = True
                reason = f"MATCH: V1 '{v1_decision}' -> V2 '{v2_action.value}'"
            else:
                match = False
                reason = f"MISMATCH: V1 '{v1_decision}' expected '{expected_v2_action.value}', got '{v2_action.value}'"

            results.append(
                RegressionResult(
                    scenario=scenario.name,
                    v1_decision=v1_decision,
                    v2_action=v2_action,
                    match=match,
                    reason=reason,
                )
            )

    return results
