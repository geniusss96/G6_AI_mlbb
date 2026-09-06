"""Unit tests for ClaudeBrainV2 (Stage 14E: V2 Brain Composition).

Verifies contract TacticalState -> ClaudeBrainV2.decide() -> Action,
regression mapping matrix for all 14 V1 decisions, dependency injection,
direction and target ID preservation, determinism, and zero hardware coupling.
"""

import random
import unittest
from actions.models import Action, ActionType
from brain.brain import ClaudeBrainV2
from brain.memory import BrainMemory
from brain.q_learning import QLearningCore
from brain.rewards import Reward, RewardCalculator
from brain.state import TacticalState
from world.models import Vector2


def _create_tactical_state(
    timestamp: float = 100.0,
    player_pos: Vector2 = Vector2(772.0, 360.0),
    hp_ratio: float = 1.0,
    is_dead: bool = False,
    s1_ready: bool = True,
    s2_ready: bool = True,
    ult_ready: bool = True,
    can_combo: bool = True,
    enemy_visible: bool = False,
    enemy_count: int = 0,
    enemy_dist: float = None,
    enemy_pos: Vector2 = None,
    enemy_id: int = None,
    minion_count: int = 0,
    minion_dist: float = None,
    minion_pos: Vector2 = None,
    minion_id: int = None,
    turret_near: bool = False,
    near_base: bool = False,
) -> TacticalState:
    return TacticalState(
        timestamp=timestamp,
        player_position=player_pos,
        player_hp_ratio=hp_ratio,
        is_player_dead=is_dead,
        s1_ready=s1_ready,
        s2_ready=s2_ready,
        ultimate_ready=ult_ready,
        can_combo=can_combo,
        enemy_visible=enemy_visible,
        enemy_count=enemy_count,
        nearest_enemy_distance=enemy_dist,
        nearest_enemy_position=enemy_pos,
        minion_count=minion_count,
        nearest_minion_distance=minion_dist,
        nearest_minion_position=minion_pos,
        enemy_near_turret=turret_near,
        is_near_base=near_base,
        nearest_enemy_id=enemy_id,
        nearest_minion_id=minion_id,
    )


class TestClaudeBrainV2(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = ClaudeBrainV2()

    def test_decision_mapping_regression_matrix_all_14_decisions(self) -> None:
        """Verifies regression matrix of all 14 V1 decision strings into typed Actions."""
        expected_matrix = {
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

        self.assertEqual(len(expected_matrix), 14, "Must cover exactly 14 decisions from inventory")

        ts = _create_tactical_state(
            player_pos=Vector2(500.0, 300.0),
            enemy_pos=Vector2(600.0, 300.0),
            enemy_id=42,
            minion_pos=Vector2(550.0, 300.0),
            minion_id=84,
        )

        for decision_str, expected_action_type in expected_matrix.items():
            action = self.brain.map_decision_to_action(decision_str, ts)
            self.assertIsInstance(action, Action)
            self.assertEqual(
                action.type,
                expected_action_type,
                f"Decision '{decision_str}' must map to {expected_action_type}, got {action.type}",
            )
            self.assertNotEqual(action.type, ActionType.IDLE)

    def test_target_id_and_direction_preservation(self) -> None:
        """Verifies target_id and directional vectors are properly preserved in Actions."""
        ts = _create_tactical_state(
            player_pos=Vector2(500.0, 300.0),
            enemy_visible=True,
            enemy_count=1,
            enemy_dist=200.0,
            enemy_pos=Vector2(700.0, 300.0),
            enemy_id=101,
        )

        action = self.brain.map_decision_to_action("KITE_AND_POKE", ts)
        self.assertEqual(action.type, ActionType.KITE)
        self.assertEqual(action.target_id, 101)
        self.assertEqual(action.direction, Vector2(700.0, 300.0))

        # Test farm action preserves minion target_id
        ts_farm = _create_tactical_state(
            player_pos=Vector2(500.0, 300.0),
            minion_count=1,
            minion_dist=200.0,
            minion_pos=Vector2(550.0, 300.0),
            minion_id=202,
        )
        action_farm = self.brain.map_decision_to_action("FARM_APPROACH", ts_farm)
        self.assertEqual(action_farm.type, ActionType.MOVE)
        self.assertEqual(action_farm.target_id, 202)
        self.assertEqual(action_farm.direction, Vector2(550.0, 300.0))

    def test_dead_player_yields_idle(self) -> None:
        """When player is dead, brain must emit ActionType.IDLE."""
        ts = _create_tactical_state(is_dead=True)
        action = self.brain.decide(ts)
        self.assertEqual(action.type, ActionType.IDLE)

    def test_turret_danger_override(self) -> None:
        """Enemy near turret triggers hard TURRET_RETREAT override."""
        ts = _create_tactical_state(
            enemy_visible=True,
            enemy_dist=250.0,
            turret_near=True,
            enemy_pos=Vector2(700.0, 300.0),
        )
        action = self.brain.decide(ts)
        self.assertEqual(action.type, ActionType.RETREAT)

    def test_critical_hp_retreat_override(self) -> None:
        """HP ratio < 0.32 triggers hard TACTICAL_RETREAT override."""
        ts = _create_tactical_state(
            enemy_visible=True,
            enemy_dist=250.0,
            hp_ratio=0.25,
            enemy_pos=Vector2(700.0, 300.0),
        )
        action = self.brain.decide(ts)
        self.assertEqual(action.type, ActionType.RETREAT)

    def test_deterministic_output_with_fixed_seed(self) -> None:
        """Brain with fixed random seed produces identical decisions for identical states."""
        b1 = ClaudeBrainV2(random_generator=random.Random(12345))
        b2 = ClaudeBrainV2(random_generator=random.Random(12345))

        ts = _create_tactical_state(
            enemy_visible=False,
            minion_count=0,
            timestamp=50.0,
        )

        a1 = b1.decide(ts)
        a2 = b2.decide(ts)
        self.assertEqual(a1.type, a2.type)
        self.assertEqual(a1.direction, a2.direction)

    def test_dependency_injection(self) -> None:
        """Verifies custom injected QLearningCore, RewardCalculator, and Memory work correctly."""
        custom_q = QLearningCore(learning_rate=0.50)
        custom_rewards = RewardCalculator()
        custom_memory = BrainMemory()

        brain = ClaudeBrainV2(
            q_learning=custom_q,
            rewards=custom_rewards,
            memory=custom_memory,
        )

        self.assertIs(brain.q_learning, custom_q)
        self.assertIs(brain.rewards, custom_rewards)
        self.assertIs(brain.memory, custom_memory)

        # Apply reward through brain
        reward = custom_rewards.for_burst_combo()
        new_q = brain.apply_reward(
            domain="combat",
            state="SWEET_SPOT_HIGH_COMBO_True",
            action="DIVE_ALL_IN",
            reward=reward,
        )
        # Verify custom Q-table was updated
        self.assertEqual(custom_q.combat_q_table["SWEET_SPOT_HIGH_COMBO_True"]["DIVE_ALL_IN"], new_q)

    def test_q_learning_core_is_used_not_duplicated(self) -> None:
        """Verifies Q-table updates reflect in brain decision exploitation."""
        q_core = QLearningCore()
        # Prime DIVE_ALL_IN with highest Q-value
        q_core.set_q("combat", "SWEET_SPOT_HIGH_COMBO_True", "KITE_AND_POKE", 10.0)
        q_core.set_q("combat", "SWEET_SPOT_HIGH_COMBO_True", "SWEET_SPOT_BURST", 20.0)
        q_core.set_q("combat", "SWEET_SPOT_HIGH_COMBO_True", "DIVE_ALL_IN", 99.0)

        # Mock RNG to force exploitation (random() >= 0.08)
        class ExploitRNG:
            def random(self) -> float:
                return 0.99

            def choice(self, seq):
                return seq[0]

        brain = ClaudeBrainV2(q_learning=q_core, random_generator=ExploitRNG())
        ts = _create_tactical_state(
            enemy_visible=True,
            enemy_dist=350.0,
            hp_ratio=0.90,
            can_combo=True,
            enemy_pos=Vector2(800.0, 300.0),
        )

        action = brain.decide(ts)
        self.assertEqual(action.type, ActionType.CAST_ULT)


if __name__ == "__main__":
    unittest.main()
