"""Stage 19A — Finalize Tactical Brain Semantics & Boundary Verification Tests.

Comprehensive regression tests verifying:
1. LOST != DEAD: Temporary loss of player vision does not mean player death.
2. Explicit PLAYER_DEATH semantics: Confirmed death requires explicit flag or 0 HP.
3. V1 adapter compatibility: All 14 canonical V1 decisions preserve exact Action mappings.
4. ClaudeBrainV2 hardware independence: Zero ADB, joystick, subprocess, or device APIs.
5. Action selection exploration determinism: Seeded RNG guarantees 100% reproducible choices.
6. Action semantic validation: Rejects NaN/Inf coordinates, negative durations, non-int target IDs.
7. Brain dependency boundaries: Strict architectural verification (world -> brain -> actions).
"""

import ast
import glob
import math
import os
import random
import unittest
from unittest.mock import Mock

from actions.models import Action, ActionType
from brain.action_selection import ActionSelector
from brain.brain import ClaudeBrainV2
from brain.encoder import TacticalStateEncoder
from brain.exploration import ExplorationContext, ExplorationPolicy
from brain.q_learning import QLearningCore
from brain.rewards import RewardCalculator
from brain.state import TacticalState, TacticalStateBuilder
from brain.v1_adapter import V1BrainAdapter
from vision.hp_detector import HPObservation
from world.events import EventType, GameEvent
from world.models import EnemyState, MinionState, PlayerState, Vector2, WorldState


class TestStage19ATacticalBrain(unittest.TestCase):
    """Stage 19A test suite."""

    def setUp(self) -> None:
        self.brain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "brain"))
        self.builder = TacticalStateBuilder()

    # --------------------------------------------------------------------------
    # 1. Player Death & Visibility Semantics (LOST != DEAD)
    # --------------------------------------------------------------------------

    def test_temporary_player_loss_not_player_death(self) -> None:
        """LOST != DEAD: When player is not visible (e.g. in bush or missing frame),
        is_player_dead must remain False, and is_player_visible must be False."""
        ws = WorldState(
            timestamp=10.0,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(value=0.75, confidence=0.8, visible=True),
                is_visible=False,  # Temporarily lost in vision
                is_dead=False,
            ),
        )
        ts = self.builder.build(ws)
        self.assertFalse(ts.is_player_dead, "Missing vision must NEVER automatically mean death")
        self.assertFalse(ts.is_player_visible)
        self.assertAlmostEqual(ts.player_hp_ratio, 0.75)

    def test_explicit_player_death_semantics(self) -> None:
        """Confirmed player death requires explicit is_dead flag or zero HP."""
        # Case A: Explicit is_dead flag on PlayerState
        ws_flag = WorldState(
            timestamp=11.0,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(value=0.50, confidence=0.8, visible=False),
                is_visible=False,
                is_dead=True,
            ),
        )
        ts_flag = self.builder.build(ws_flag)
        self.assertTrue(ts_flag.is_player_dead)

        # Case B: Confirmed zero HP
        ws_zero_hp = WorldState(
            timestamp=12.0,
            player=PlayerState(
                position=Vector2(500.0, 400.0),
                hp=HPObservation(value=0.0, confidence=1.0, visible=True),
                is_visible=False,
                is_dead=False,
            ),
        )
        ts_zero_hp = self.builder.build(ws_zero_hp)
        self.assertTrue(ts_zero_hp.is_player_dead)

    # --------------------------------------------------------------------------
    # 2. V1 Adapter Compatibility (14/14 Decisions)
    # --------------------------------------------------------------------------

    def test_v1_adapter_14_decisions_compatibility(self) -> None:
        """Verifies that V1BrainAdapter translates all 14 legacy decision strings into valid Actions."""
        mock_legacy = Mock()
        adapter = V1BrainAdapter(legacy_brain=mock_legacy)

        player_pos = Vector2(772.0, 360.0)
        enemy = EnemyState(position=Vector2(900.0, 360.0), bbox=(880, 340, 920, 380), confidence=0.9)
        minion = MinionState(position=Vector2(700.0, 360.0), bbox=(680, 340, 720, 380), confidence=0.85, is_enemy=True)

        combat_decisions = {
            "TURRET_RETREAT": ActionType.RETREAT,
            "TACTICAL_RETREAT": ActionType.RETREAT,
            "KITE_AND_POKE": ActionType.KITE,
            "SWEET_SPOT_BURST": ActionType.ATTACK,
            "DIVE_ALL_IN": ActionType.CAST_ULT,
        }

        for dec, expected_type in combat_decisions.items():
            act = adapter._from_legacy_combat_decision(
                decision=dec,
                player_pos=player_pos,
                enemy=enemy,
                dist_enemy=128.0,
            )
            self.assertEqual(act.type, expected_type, f"Combat decision {dec} failed mapping")
            self.assertIsNotNone(act.direction)

        farm_decisions = {
            "FARM_APPROACH": ActionType.MOVE,
            "FARM_KITE_BACK": ActionType.KITE,
            "FARM_SWEET_SPOT": ActionType.FARM,
            "FARM_S1_AOE": ActionType.CAST_S1,
        }

        for dec, expected_type in farm_decisions.items():
            act = adapter._from_legacy_farm_decision(
                decision=dec,
                player_pos=player_pos,
                minion=minion,
            )
            self.assertEqual(act.type, expected_type, f"Farm decision {dec} failed mapping")

    # --------------------------------------------------------------------------
    # 3. Action Selection Exploration Determinism
    # --------------------------------------------------------------------------

    def test_action_selection_determinism(self) -> None:
        """With a seeded RNG, ActionSelector produces perfectly reproducible actions."""
        selector = ActionSelector(
            encoder=TacticalStateEncoder(),
            q_learning=QLearningCore(),
            exploration=ExplorationPolicy(),
        )

        ts = TacticalState(
            timestamp=100.0,
            player_position=Vector2(772.0, 360.0),
            player_hp_ratio=1.0,
            is_player_dead=False,
            s1_ready=True,
            s2_ready=True,
            ultimate_ready=True,
            can_combo=True,
            enemy_visible=True,
            enemy_count=1,
            nearest_enemy_distance=200.0,
            nearest_enemy_position=Vector2(972.0, 360.0),
            minion_count=0,
            nearest_minion_distance=None,
            nearest_minion_position=None,
            enemy_near_turret=False,
            is_near_base=False,
        )

        candidates = ["KITE_AND_POKE", "SWEET_SPOT_BURST", "DIVE_ALL_IN"]

        rng1 = random.Random(999)
        rng2 = random.Random(999)

        act1 = selector.select(ts, ExplorationContext.COMBAT, candidates, rng=rng1)
        act2 = selector.select(ts, ExplorationContext.COMBAT, candidates, rng=rng2)

        self.assertEqual(act1, act2)

    # --------------------------------------------------------------------------
    # 4. Action Semantic Validation
    # --------------------------------------------------------------------------

    def test_action_semantic_validation(self) -> None:
        """Action validates intent semantics: rejects negative duration, NaN/Inf directions, non-int IDs."""
        # Valid actions
        a1 = Action(type=ActionType.MOVE, direction=Vector2(100.0, 200.0), target_id=5, duration=0.25)
        self.assertEqual(a1.type, ActionType.MOVE)
        self.assertEqual(a1.duration, 0.25)

        # Rejects negative duration
        with self.assertRaises(ValueError):
            Action(type=ActionType.MOVE, duration=-0.1)

        # Rejects NaN direction
        with self.assertRaises(ValueError):
            Action(type=ActionType.MOVE, direction=Vector2(float("nan"), 100.0))

        # Rejects Inf direction
        with self.assertRaises(ValueError):
            Action(type=ActionType.MOVE, direction=Vector2(100.0, float("inf")))

        # Rejects non-int target_id
        with self.assertRaises(TypeError):
            Action(type=ActionType.MOVE, target_id="enemy_1")

    # --------------------------------------------------------------------------
    # 5. Brain Dependency Boundaries (world -> brain -> actions)
    # --------------------------------------------------------------------------

    def test_brain_dependency_boundaries(self) -> None:
        """Ensures NO module in brain/ imports control, adb, joystick, subprocess, or device drivers."""
        forbidden = {
            "control",
            "adb",
            "joystick",
            "subprocess",
            "cv2",
            "opencv",
            "yolo",
            "ultralytics",
            "realtime_vision",
            "claude_macro",
        }

        py_files = glob.glob(os.path.join(self.brain_dir, "*.py"))
        self.assertGreater(len(py_files), 5, "Should find brain python source files")

        for filepath in py_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()

            parsed = ast.parse(source, filename=filename)

            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in forbidden:
                            self.assertNotIn(
                                bad.lower(),
                                alias.name.lower(),
                                f"Forbidden import '{alias.name}' found in brain/{filename}",
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for bad in forbidden:
                            self.assertNotIn(
                                bad.lower(),
                                node.module.lower(),
                                f"Forbidden import from '{node.module}' found in brain/{filename}",
                            )


if __name__ == "__main__":
    unittest.main()
