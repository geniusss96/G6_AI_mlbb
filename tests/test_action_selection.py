"""Tests for brain.action_selection (Claude Tactical AI V2).

Verifies:
- Pure exploitation selects argmax_a Q(s, a)
- Exploration branch uses ExplorationPolicy.should_explore
- Exact V1 epsilon probabilities preserved (ROAM=0.20, FARM=0.10, COMBAT=0.08)
- Candidate actions strictly constrain selection
- Deterministic RNG produces identical results
- Rejection of invalid exploration contexts and empty candidate action sets
- Zero Q-table mutations during action selection
- Complete isolation from memory, rewards, hardware, and device controllers
- Complete mapping of V1 decision strings to ActionType
"""

import ast
import copy
import os
import random
import unittest
from unittest.mock import Mock, patch

from actions.models import Action, ActionType
from brain.action_selection import ActionSelector
from brain.encoder import TacticalStateEncoder
from brain.exploration import ExplorationContext, ExplorationPolicy
from brain.q_learning import QLearningCore
from brain.state import TacticalState
from world.models import Vector2


class DeterministicRNG:
    """Mock RNG to deterministically force exploration or exploitation."""

    def __init__(self, random_val: float = 0.99, choice_index: int = 0) -> None:
        self.random_val = random_val
        self.choice_index = choice_index

    def random(self) -> float:
        return self.random_val

    def choice(self, seq: list) -> any:
        return seq[self.choice_index % len(seq)]


def _make_state(
    timestamp: float = 100.0,
    player_pos: Vector2 = Vector2(772.0, 360.0),
    hp_ratio: float = 1.0,
    enemy_visible: bool = True,
    enemy_dist: float = 200.0,
    can_combo: bool = True,
    s1_ready: bool = True,
    s2_ready: bool = True,
    ultimate_ready: bool = True,
) -> TacticalState:
    """Constructs a deterministic TacticalState."""
    return TacticalState(
        timestamp=timestamp,
        player_position=player_pos,
        player_hp_ratio=hp_ratio,
        is_player_dead=False,
        s1_ready=s1_ready,
        s2_ready=s2_ready,
        ultimate_ready=ultimate_ready,
        can_combo=can_combo,
        enemy_visible=enemy_visible,
        enemy_count=1 if enemy_visible else 0,
        nearest_enemy_distance=enemy_dist if enemy_visible else None,
        nearest_enemy_position=Vector2(player_pos.x + enemy_dist, player_pos.y) if enemy_visible else None,
        minion_count=0,
        nearest_minion_distance=None,
        nearest_minion_position=None,
        enemy_near_turret=False,
        is_near_base=False,
        nearest_enemy_id=1 if enemy_visible else None,
        nearest_minion_id=None,
    )


class TestActionSelection(unittest.TestCase):
    """Test suite for ActionSelector."""

    def setUp(self) -> None:
        self.encoder = TacticalStateEncoder()
        self.q_learning = QLearningCore()
        self.exploration = ExplorationPolicy()
        self.selector = ActionSelector(
            encoder=self.encoder,
            q_learning=self.q_learning,
            exploration=self.exploration,
        )

    def test_exploitation_selects_max_q(self) -> None:
        """Exploitation (random() > epsilon) selects action with maximum Q-value."""
        state = _make_state(enemy_dist=200.0)  # Encodes to DANGER_CLOSE_HIGH_COMBO_True
        state_key = self.encoder.encode(state, context="combat")

        # Set distinct Q-values
        self.q_learning.set_q("combat", state_key, "KITE_AND_POKE", 10.0)
        self.q_learning.set_q("combat", state_key, "SWEET_SPOT_BURST", 35.0)
        self.q_learning.set_q("combat", state_key, "DIVE_ALL_IN", 5.0)

        # RNG returning 0.99 forces exploitation (epsilon=0.08 < 0.99)
        rng = DeterministicRNG(random_val=0.99)
        candidates = ["KITE_AND_POKE", "SWEET_SPOT_BURST", "DIVE_ALL_IN"]

        action = self.selector.select(
            state=state,
            context=ExplorationContext.COMBAT,
            candidate_actions=candidates,
            rng=rng,
        )

        self.assertIsInstance(action, Action)
        self.assertEqual(action.type, ActionType.ATTACK)  # SWEET_SPOT_BURST -> ATTACK

    def test_exploration_uses_exploration_policy(self) -> None:
        """Exploration (random() < epsilon) chooses a candidate action via RNG choice."""
        state = _make_state(enemy_dist=200.0)
        candidates = ["KITE_AND_POKE", "SWEET_SPOT_BURST", "DIVE_ALL_IN"]

        # RNG returning 0.0 forces exploration (0.0 < 0.08) and picks index 2 ("DIVE_ALL_IN")
        rng = DeterministicRNG(random_val=0.0, choice_index=2)

        with patch.object(self.exploration, "should_explore", wraps=self.exploration.should_explore) as mock_exp:
            action = self.selector.select(
                state=state,
                context=ExplorationContext.COMBAT,
                candidate_actions=candidates,
                rng=rng,
            )
            mock_exp.assert_called_once_with(ExplorationContext.COMBAT, rng=rng)

        self.assertEqual(action.type, ActionType.CAST_ULT)  # DIVE_ALL_IN -> CAST_ULT

    def test_context_default_epsilons(self) -> None:
        """Verifies exact V1 default epsilon probabilities across contexts."""
        self.assertAlmostEqual(self.selector.exploration.epsilon(ExplorationContext.ROAM), 0.20)
        self.assertAlmostEqual(self.selector.exploration.epsilon(ExplorationContext.FARM), 0.10)
        self.assertAlmostEqual(self.selector.exploration.epsilon(ExplorationContext.COMBAT), 0.08)

    def test_candidate_actions_constrain_selection(self) -> None:
        """Selection is strictly confined to candidate_actions even if higher Q exists outside."""
        state = _make_state(enemy_dist=200.0)
        state_key = self.encoder.encode(state, context="combat")

        # Set huge Q on an unpermitted action
        self.q_learning.set_q("combat", state_key, "FORBIDDEN_MEGA_ACTION", 999.0)
        self.q_learning.set_q("combat", state_key, "KITE_AND_POKE", 10.0)
        self.q_learning.set_q("combat", state_key, "DIVE_ALL_IN", 20.0)

        # Only allow KITE_AND_POKE and DIVE_ALL_IN
        allowed = ["KITE_AND_POKE", "DIVE_ALL_IN"]
        rng = DeterministicRNG(random_val=0.99)

        action = self.selector.select(
            state=state,
            context=ExplorationContext.COMBAT,
            candidate_actions=allowed,
            rng=rng,
        )

        self.assertEqual(action.type, ActionType.CAST_ULT)  # DIVE_ALL_IN selected, not FORBIDDEN

    def test_deterministic_rng_reproducibility(self) -> None:
        """Identical inputs and seeded RNG produce identical decisions."""
        state = _make_state()
        candidates = ["FARM_APPROACH", "FARM_KITE_BACK", "FARM_SWEET_SPOT"]

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        act1 = self.selector.select(state, ExplorationContext.FARM, candidates, rng=rng1)
        act2 = self.selector.select(state, ExplorationContext.FARM, candidates, rng=rng2)

        self.assertEqual(act1, act2)

    def test_invalid_context_rejected(self) -> None:
        """Passing unknown or invalid context raises ValueError."""
        state = _make_state()
        with self.assertRaises(ValueError):
            self.selector.select(state, "INVALID_CONTEXT", ["KITE_AND_POKE"])

    def test_empty_candidate_set_rejected(self) -> None:
        """Passing empty candidate_actions raises ValueError."""
        state = _make_state()
        with self.assertRaises(ValueError):
            self.selector.select(state, ExplorationContext.COMBAT, [])

    def test_no_q_table_mutation(self) -> None:
        """Action selection does not mutate Q-tables in any way."""
        state = _make_state()
        state_key = self.encoder.encode(state, context="combat")
        self.q_learning.set_q("combat", state_key, "KITE_AND_POKE", 5.0)

        snapshot_before = copy.deepcopy(self.q_learning.combat_q_table)
        self.selector.select(state, ExplorationContext.COMBAT, ["KITE_AND_POKE", "DIVE_ALL_IN"])
        snapshot_after = copy.deepcopy(self.q_learning.combat_q_table)

        self.assertEqual(snapshot_before, snapshot_after)

    def test_no_hardware_dependencies(self) -> None:
        """Ensures brain/action_selection.py has no hardware/vision/control imports."""
        module_path = os.path.join(
            os.path.dirname(__file__), "..", "brain", "action_selection.py"
        )
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()

        parsed = ast.parse(source)
        forbidden = {
            "control", "adb", "joystick", "opencv", "cv2", "yolo",
            "ultralytics", "realtime_vision", "ActionExecutor",
            "RewardCalculator", "LearningLoop",
        }

        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in forbidden:
                        self.assertNotIn(bad.lower(), alias.name.lower())
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for bad in forbidden:
                        self.assertNotIn(bad.lower(), node.module.lower())

    def test_v1_decision_to_action_type_mapping(self) -> None:
        """Verifies mapping of all 14 canonical V1 decision strings to ActionType."""
        expected_mappings = {
            "TURRET_RETREAT": ActionType.RETREAT,
            "TACTICAL_RETREAT": ActionType.RETREAT,
            "KITE_AND_POKE": ActionType.KITE,
            "SWEET_SPOT_BURST": ActionType.ATTACK,
            "DIVE_ALL_IN": ActionType.CAST_ULT,
            "FARM_APPROACH": ActionType.MOVE,
            "FARM_KITE_BACK": ActionType.KITE,
            "FARM_SWEET_SPOT": ActionType.FARM,
            "FARM_S1_AOE": ActionType.CAST_S1,
            "LANE_ADVANCE": ActionType.MOVE,
            "LANE_PUSH": ActionType.MOVE,
            "RIVER_SCOUT": ActionType.SCOUT,
            "FLANK_ADVANCE": ActionType.SCOUT,
            "CREEP_INTERCEPT": ActionType.FARM,
        }

        for decision_str, expected_type in expected_mappings.items():
            mapped = self.selector.to_action_type(decision_str)
            self.assertEqual(
                mapped,
                expected_type,
                f"Expected {decision_str} -> {expected_type}, got {mapped}",
            )


if __name__ == "__main__":
    unittest.main()
