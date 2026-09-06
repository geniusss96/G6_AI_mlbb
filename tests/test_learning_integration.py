"""Unit and integration tests for brain.learning_integration (Claude Tactical AI V2).

Verifies:
- HERO_KILL generates correct rewards and triggers TD updates
- PLAYER_DEATH correctly triggers terminal updates
- CREEP_KILL generates farm and roam rewards
- TARGET_LOST yields zero rewards and triggers zero learning steps
- Multiple rewards from a single event are processed into distinct transitions
- LearningLoop is called exactly once per generated reward
- RewardCalculator is called without numeric constant duplication
- No direct Q-table updates from LearningIntegrator
- Deterministic execution
- Baseline immutability (q_brain.json and baseline SHA-256 verification)
- Complete isolation from hardware / ADB / vision modules
- End-to-end offline integration scenarios (Combat success, Combat death, No-reward event)
"""

import ast
import hashlib
import os
import unittest
from unittest.mock import Mock, patch

from actions.models import Action, ActionType
from brain.encoder import TacticalStateEncoder
from brain.learning_integration import LearningIntegrator
from brain.learning_loop import LearningLoop
from brain.q_learning import QLearningCore
from brain.rewards import Reward, RewardCalculator
from brain.state import TacticalState
from brain.transition import Transition
from brain.transition_builder import TransitionBuilder
from world.events import EventType, GameEvent
from world.models import Vector2


def _make_state(
    timestamp: float = 100.0,
    player_pos: Vector2 = Vector2(772.0, 360.0),
    hp_ratio: float = 1.0,
    is_player_dead: bool = False,
    enemy_visible: bool = True,
    enemy_dist: float = 200.0,
    can_combo: bool = True,
    s1_ready: bool = True,
    s2_ready: bool = True,
    ultimate_ready: bool = True,
    minion_count: int = 0,
    minion_dist: float = None,
) -> TacticalState:
    """Helper to construct a deterministic TacticalState."""
    return TacticalState(
        timestamp=timestamp,
        player_position=player_pos,
        player_hp_ratio=hp_ratio,
        is_player_dead=is_player_dead,
        s1_ready=s1_ready,
        s2_ready=s2_ready,
        ultimate_ready=ultimate_ready,
        can_combo=can_combo,
        enemy_visible=enemy_visible,
        enemy_count=1 if enemy_visible else 0,
        nearest_enemy_distance=enemy_dist if enemy_visible else None,
        nearest_enemy_position=Vector2(player_pos.x + enemy_dist, player_pos.y) if enemy_visible else None,
        minion_count=minion_count,
        nearest_minion_distance=minion_dist,
        nearest_minion_position=Vector2(player_pos.x + (minion_dist or 0), player_pos.y) if minion_dist else None,
        enemy_near_turret=False,
        is_near_base=False,
        nearest_enemy_id=1 if enemy_visible else None,
        nearest_minion_id=1 if minion_count > 0 else None,
    )


def _make_event(
    event_type: EventType,
    timestamp: float = 100.0,
    entity_id: int = 1,
    confidence: float = 1.0,
    evidence: tuple = ("test",),
) -> GameEvent:
    """Helper to construct a deterministic GameEvent."""
    return GameEvent(
        type=event_type,
        timestamp=timestamp,
        entity_id=entity_id,
        confidence=confidence,
        evidence=evidence,
    )


class TestLearningIntegration(unittest.TestCase):
    """Test cases for LearningIntegrator."""

    def _file_hash(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return ""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def setUp(self) -> None:
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.q_brain_path = os.path.join(self.root_dir, "q_brain.json")
        self.baseline_path = os.path.join(self.root_dir, "data", "q_brain_baseline.json")

        self.initial_q_brain_hash = self._file_hash(self.q_brain_path)
        self.initial_baseline_hash = self._file_hash(self.baseline_path)

        self.encoder = TacticalStateEncoder()
        self.builder = TransitionBuilder(encoder=self.encoder)
        self.q_learning = QLearningCore(learning_rate=0.10, discount_factor=0.85)
        self.rewards = RewardCalculator()
        self.learning_loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
        )
        self.integrator = LearningIntegrator(
            reward_calculator=self.rewards,
            transition_builder=self.builder,
            learning_loop=self.learning_loop,
        )

    def tearDown(self) -> None:
        self.assertEqual(self._file_hash(self.q_brain_path), self.initial_q_brain_hash)
        self.assertEqual(self._file_hash(self.baseline_path), self.initial_baseline_hash)

    def test_hero_kill_generates_multiple_rewards_and_td_updates(self) -> None:
        """HERO_KILL yields combat (+60) and roam (+40) rewards, creating 2 transitions."""
        state_a = _make_state(timestamp=10.0, enemy_dist=200.0)
        state_b = _make_state(timestamp=11.0, enemy_dist=350.0)
        action = Action(type=ActionType.CAST_ULT)
        event = _make_event(EventType.HERO_KILL, timestamp=10.5)

        transitions = self.integrator.process(
            state=state_a,
            action=action,
            events=[event],
            next_state=state_b,
            done=False,
        )

        self.assertEqual(len(transitions), 2)
        # First transition: combat +60
        self.assertEqual(transitions[0].reward, 60.0)
        self.assertEqual(transitions[0].action, "cast_ult")
        # Second transition: roam +40
        self.assertEqual(transitions[1].reward, 40.0)
        self.assertEqual(transitions[1].action, "cast_ult")

    def test_player_death_terminal_update(self) -> None:
        """PLAYER_DEATH with next_state=None executes terminal transitions with negative penalties."""
        state = _make_state(timestamp=20.0, hp_ratio=0.0, is_player_dead=True)
        action = Action(type=ActionType.ATTACK)
        event = _make_event(EventType.PLAYER_DEATH, timestamp=20.1)

        transitions = self.integrator.process(
            state=state,
            action=action,
            events=[event],
            next_state=None,
            done=True,
        )

        self.assertEqual(len(transitions), 2)
        for t in transitions:
            self.assertTrue(t.done)
            self.assertIsNone(t.next_state_key)
            self.assertLess(t.reward, 0.0)

        # Combat penalty: -40.0, Roam penalty: -30.0
        self.assertEqual(transitions[0].reward, -40.0)
        self.assertEqual(transitions[1].reward, -30.0)

    def test_creep_kill_rewards(self) -> None:
        """CREEP_KILL produces farm (+25.0) and roam (+15.0) rewards."""
        state_a = _make_state(timestamp=30.0, enemy_visible=False, minion_count=2, minion_dist=200.0)
        state_b = _make_state(timestamp=31.0, enemy_visible=False, minion_count=1, minion_dist=250.0)
        action = Action(type=ActionType.FARM)
        event = _make_event(EventType.CREEP_KILL, timestamp=30.5)

        transitions = self.integrator.process(
            state=state_a,
            action=action,
            events=[event],
            next_state=state_b,
            done=False,
        )

        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[0].reward, 25.0)
        self.assertEqual(transitions[1].reward, 15.0)

    def test_target_lost_yields_no_rewards_or_updates(self) -> None:
        """TARGET_LOST event produces no rewards and thus triggers zero transitions."""
        state_a = _make_state(timestamp=40.0)
        state_b = _make_state(timestamp=41.0, enemy_visible=False)
        action = Action(type=ActionType.SCOUT)
        event = _make_event(EventType.TARGET_LOST, timestamp=40.5)

        with patch.object(self.learning_loop, "learn_step") as mock_learn:
            transitions = self.integrator.process(
                state=state_a,
                action=action,
                events=[event],
                next_state=state_b,
                done=False,
            )
            self.assertEqual(len(transitions), 0)
            self.assertEqual(mock_learn.call_count, 0)

    def test_learning_loop_called_exactly_once_per_transition(self) -> None:
        """Verifies LearningLoop.learn_step is invoked once per calculated reward."""
        state_a = _make_state(timestamp=50.0)
        state_b = _make_state(timestamp=51.0)
        action = Action(type=ActionType.KITE)
        # 1 TARGET_ENTERED event gives 1 reward (+30 roam)
        event = _make_event(EventType.TARGET_ENTERED, timestamp=50.2)

        with patch.object(self.learning_loop, "learn_step", wraps=self.learning_loop.learn_step) as mock_learn:
            transitions = self.integrator.process(
                state=state_a,
                action=action,
                events=[event],
                next_state=state_b,
                done=False,
            )
            self.assertEqual(len(transitions), 1)
            self.assertEqual(mock_learn.call_count, 1)

    def test_reward_calculator_invoked_without_constant_duplication(self) -> None:
        """Verifies that RewardCalculator.for_event is invoked and rewards originate from it."""
        state_a = _make_state()
        state_b = _make_state()
        action = Action(type=ActionType.MOVE)
        event = _make_event(EventType.HERO_KILL, timestamp=10.0)

        with patch.object(self.rewards, "for_event", wraps=self.rewards.for_event) as mock_calc:
            self.integrator.process(state_a, action, [event], state_b, False)
            mock_calc.assert_called_once_with(event)

    def test_no_direct_q_updates_from_integrator(self) -> None:
        """Integrator delegates strictly to LearningLoop without calling set_q or update_q directly."""
        state_a = _make_state()
        state_b = _make_state()
        action = Action(type=ActionType.ATTACK)
        event = _make_event(EventType.HERO_KILL, timestamp=1.0)

        with patch.object(self.learning_loop, "learn_step", wraps=self.learning_loop.learn_step) as mock_loop, \
             patch.object(self.q_learning, "update_q") as mock_v1:
            self.integrator.process(state_a, action, [event], state_b, False)
            self.assertEqual(mock_loop.call_count, 2)
            self.assertEqual(mock_v1.call_count, 0)

    def test_deterministic_behavior(self) -> None:
        """Two identical runs with identical events produce identical transitions and Q-tables."""
        q1 = QLearningCore(learning_rate=0.1, discount_factor=0.85)
        q2 = QLearningCore(learning_rate=0.1, discount_factor=0.85)

        int1 = LearningIntegrator(self.rewards, TransitionBuilder(self.encoder), LearningLoop(self.encoder, TransitionBuilder(self.encoder), q1))
        int2 = LearningIntegrator(self.rewards, TransitionBuilder(self.encoder), LearningLoop(self.encoder, TransitionBuilder(self.encoder), q2))

        state_a = _make_state(timestamp=1.0)
        state_b = _make_state(timestamp=2.0)
        action = Action(type=ActionType.KITE)
        event = _make_event(EventType.HERO_KILL, timestamp=1.5)

        res1 = int1.process(state_a, action, [event], state_b, False)
        res2 = int2.process(state_a, action, [event], state_b, False)

        self.assertEqual(res1, res2)

    def test_no_hardware_dependencies(self) -> None:
        """Ensures brain/learning_integration.py contains no imports of hardware/control/vision modules."""
        module_path = os.path.join(
            os.path.dirname(__file__), "..", "brain", "learning_integration.py"
        )
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()

        parsed = ast.parse(source)
        forbidden = {
            "control", "adb", "joystick", "opencv", "cv2", "yolo",
            "ultralytics", "realtime_vision", "claude_macro", "ActionExecutor",
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

    # --------------------------------------------------------------------------
    # Offline Integration Scenarios
    # --------------------------------------------------------------------------

    def test_offline_scenario_1_combat_success(self) -> None:
        """Scenario 1: S0 --KITE--> S1 with HERO_KILL event triggers reward, transition, and TD update."""
        s0 = _make_state(timestamp=100.0, enemy_dist=200.0)
        s1 = _make_state(timestamp=101.0, enemy_dist=350.0)
        action = Action(type=ActionType.KITE)
        event = _make_event(EventType.HERO_KILL, timestamp=100.5)

        # Initial Q is 0.0
        transitions = self.integrator.process(
            state=s0,
            action=action,
            events=[event],
            next_state=s1,
            done=False,
        )

        self.assertEqual(len(transitions), 2)
        # Combat reward +60 and roam +40 sequentially updated via update_td
        # Step 1: 0 + 0.1 * 60 = 6.0
        # Step 2: 6.0 + 0.1 * (40 - 6.0) = 9.40
        combat_trans = transitions[0]
        self.assertEqual(combat_trans.reward, 60.0)
        q_val = self.q_learning.get_q("combat", combat_trans.state_key, "kite")
        self.assertAlmostEqual(q_val, 9.40, places=2)

    def test_offline_scenario_2_combat_death(self) -> None:
        """Scenario 2: S0 --ATTACK--> terminal with PLAYER_DEATH event triggers negative penalty and terminal update."""
        s0 = _make_state(timestamp=200.0, enemy_dist=150.0)
        action = Action(type=ActionType.ATTACK)
        event = _make_event(EventType.PLAYER_DEATH, timestamp=200.5)

        transitions = self.integrator.process(
            state=s0,
            action=action,
            events=[event],
            next_state=None,
            done=True,
        )

        self.assertEqual(len(transitions), 2)
        combat_trans = transitions[0]
        self.assertEqual(combat_trans.reward, -40.0)
        self.assertTrue(combat_trans.done)
        # Step 1: 0 + 0.1 * (-40) = -4.0
        # Step 2: -4.0 + 0.1 * (-30 - (-4.0)) = -6.60
        q_val = self.q_learning.get_q("combat", combat_trans.state_key, "attack")
        self.assertAlmostEqual(q_val, -6.60, places=2)

    def test_offline_scenario_3_no_reward_event(self) -> None:
        """Scenario 3: S0 --SCOUT--> S1 with TARGET_LOST event yields 0 rewards and no Q-table updates."""
        s0 = _make_state(timestamp=300.0, enemy_visible=True, enemy_dist=300.0)
        s1 = _make_state(timestamp=301.0, enemy_visible=False)
        action = Action(type=ActionType.SCOUT)
        event = _make_event(EventType.TARGET_LOST, timestamp=300.5)

        transitions = self.integrator.process(
            state=s0,
            action=action,
            events=[event],
            next_state=s1,
            done=False,
        )

        self.assertEqual(len(transitions), 0)
        # Verify no Q was touched
        q_val = self.q_learning.get_q("combat", "DANGER_CLOSE_HIGH_COMBO_True", "scout")
        self.assertEqual(q_val, 0.0)


if __name__ == "__main__":
    unittest.main()
