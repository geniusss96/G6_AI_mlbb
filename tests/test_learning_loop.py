"""Unit and integration tests for brain.learning_loop (Claude Tactical AI V2).

Verifies:
- Non-terminal and terminal learning steps
- Exact invocation counts for encoder, transition builder, and QLearningCore.update_td
- Absence of action selection, exploration, reward calculation, and automatic replay
- Optional memory logging in BrainMemory
- Input validation (rejection of non-terminal steps without next_state)
- Deterministic behavior
- Strict architectural isolation from hardware / vision / control modules
- End-to-end regression verifying the V2 Temporal-Difference update
"""

import ast
import os
import unittest
from unittest.mock import MagicMock, Mock, patch

from actions.models import Action, ActionType
from brain.encoder import TacticalStateEncoder
from brain.learning_loop import LearningLoop
from brain.memory import BrainMemory
from brain.q_learning import QLearningCore
from brain.replay import ExperienceReplay
from brain.state import TacticalState
from brain.transition import Transition
from brain.transition_builder import TransitionBuilder
from world.models import Vector2


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
    """Helper to construct a deterministic TacticalState."""
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


class TestLearningLoop(unittest.TestCase):
    """Unit and integration test cases for LearningLoop."""

    def setUp(self) -> None:
        self.encoder = TacticalStateEncoder()
        self.builder = TransitionBuilder(encoder=self.encoder)
        self.q_learning = QLearningCore(learning_rate=0.10, discount_factor=0.85)

    def test_non_terminal_learning_step(self) -> None:
        """Tests standard non-terminal step updating Q-table via update_td."""
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
        )

        state_a = _make_state(timestamp=10.0, enemy_dist=200.0)
        state_b = _make_state(timestamp=11.0, enemy_dist=350.0)
        action = Action(type=ActionType.KITE, direction=Vector2(-1.0, 0.0))

        transition = loop.learn_step(
            state=state_a,
            action=action,
            reward=5.0,
            next_state=state_b,
            done=False,
        )

        self.assertIsInstance(transition, Transition)
        self.assertEqual(transition.state_key, "DANGER_CLOSE_HIGH_COMBO_True")
        self.assertEqual(transition.action, "kite")
        self.assertEqual(transition.reward, 5.0)
        self.assertEqual(transition.next_state_key, "SWEET_SPOT_HIGH_COMBO_True")
        self.assertFalse(transition.done)

        # Initial Q was 0.0, next max Q was 0.0 -> target = 5.0 -> new_q = 0.50
        q_val = self.q_learning.get_q("combat", "DANGER_CLOSE_HIGH_COMBO_True", "kite")
        self.assertAlmostEqual(q_val, 0.50, places=2)

    def test_terminal_learning_step(self) -> None:
        """Tests terminal step with next_state=None and done=True."""
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
        )

        state = _make_state(timestamp=20.0, enemy_dist=200.0)
        action = Action(type=ActionType.CAST_ULT)

        transition = loop.learn_step(
            state=state,
            action=action,
            reward=60.0,
            next_state=None,
            done=True,
        )

        self.assertTrue(transition.done)
        self.assertIsNone(transition.next_state_key)
        self.assertEqual(transition.reward, 60.0)
        self.assertEqual(transition.action, "cast_ult")

        # Terminal target = 60.0 -> new_q = 0 + 0.1 * (60 - 0) = 6.0
        q_val = self.q_learning.get_q("combat", "DANGER_CLOSE_HIGH_COMBO_True", "cast_ult")
        self.assertAlmostEqual(q_val, 6.0, places=2)

    def test_encoder_invoked_exactly_once_for_terminal_step(self) -> None:
        """Tests that for a terminal step, encoder.encode is called exactly once (for state)."""
        mock_encoder = Mock(spec=TacticalStateEncoder)
        mock_encoder.encode.return_value = "MOCK_STATE_KEY"

        builder = TransitionBuilder(encoder=mock_encoder)
        loop = LearningLoop(
            encoder=mock_encoder,
            transition_builder=builder,
            q_learning=self.q_learning,
        )

        state = _make_state(timestamp=30.0)
        loop.learn_step(
            state=state,
            action=Action(type=ActionType.ATTACK),
            reward=10.0,
            next_state=None,
            done=True,
        )

        self.assertEqual(mock_encoder.encode.call_count, 1)

    def test_transition_builder_invoked(self) -> None:
        """Tests that TransitionBuilder.build is invoked with expected arguments."""
        mock_builder = Mock(spec=TransitionBuilder)
        mock_builder.build.return_value = Transition(
            state_key="S0",
            action="KITE",
            reward=5.0,
            next_state_key=None,
            done=True,
        )

        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=mock_builder,
            q_learning=self.q_learning,
        )

        state = _make_state()
        action = Action(type=ActionType.KITE)
        loop.learn_step(state=state, action=action, reward=5.0, next_state=None, done=True)

        mock_builder.build.assert_called_once_with(
            state=state,
            action=action,
            reward=5.0,
            next_state=None,
            done=True,
        )

    def test_update_td_invoked_exactly_once(self) -> None:
        """Tests that QLearningCore.update_td is called exactly once per learn_step."""
        mock_q = Mock(spec=QLearningCore)
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=mock_q,
        )

        state = _make_state()
        loop.learn_step(state=state, action="ATTACK", reward=1.0, next_state=None, done=True)

        self.assertEqual(mock_q.update_td.call_count, 1)

    def test_no_action_selection_or_exploration(self) -> None:
        """Verifies that LearningLoop does not perform action selection or call exploration."""
        mock_q = Mock(spec=QLearningCore)
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=mock_q,
        )

        state = _make_state()
        loop.learn_step(state=state, action="EXTERNAL_ACTION", reward=0.0, next_state=None, done=True)

        # get_best_action must NOT be called
        self.assertEqual(mock_q.get_best_action.call_count, 0)

    def test_memory_optional(self) -> None:
        """Tests that LearningLoop operates seamlessly when memory is None."""
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
            memory=None,
        )
        state = _make_state()
        t = loop.learn_step(state=state, action="KITE", reward=2.0, next_state=None, done=True)
        self.assertIsInstance(t, Transition)

    def test_memory_receives_experience_when_provided(self) -> None:
        """Tests that BrainMemory logs action record when provided to LearningLoop."""
        memory = BrainMemory()
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
            memory=memory,
        )

        state = _make_state(timestamp=123.45)
        loop.learn_step(state=state, action="KITE", reward=2.0, next_state=None, done=True)

        self.assertEqual(len(memory.recent_action_history), 1)
        record = memory.recent_action_history[0]
        self.assertEqual(record.action, "KITE")
        self.assertEqual(record.category, "COMBAT")
        self.assertEqual(record.state, "DANGER_CLOSE_HIGH_COMBO_True")
        self.assertEqual(record.timestamp, 123.45)

    def test_replay_not_automatically_triggered(self) -> None:
        """Verifies that ExperienceReplay is never automatically triggered during learn_step."""
        memory = BrainMemory()
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
            memory=memory,
        )

        state = _make_state()
        with patch.object(ExperienceReplay, "replay_moment") as mock_replay_moment, \
             patch.object(ExperienceReplay, "replay") as mock_replay:
            loop.learn_step(state=state, action="DIVE_ALL_IN", reward=60.0, next_state=None, done=True)
            self.assertEqual(mock_replay_moment.call_count, 0)
            self.assertEqual(mock_replay.call_count, 0)

    def test_invalid_terminal_state_rejected(self) -> None:
        """Verifies that done=False with next_state=None raises ValueError."""
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
        )
        state = _make_state()
        with self.assertRaises(ValueError):
            loop.learn_step(state=state, action="MOVE", reward=1.0, next_state=None, done=False)

    def test_deterministic_result(self) -> None:
        """Verifies determinism: two independent loops produce identical transitions and Q-tables."""
        q1 = QLearningCore(learning_rate=0.1, discount_factor=0.85)
        q2 = QLearningCore(learning_rate=0.1, discount_factor=0.85)

        loop1 = LearningLoop(self.encoder, TransitionBuilder(self.encoder), q1)
        loop2 = LearningLoop(self.encoder, TransitionBuilder(self.encoder), q2)

        state_a = _make_state(timestamp=1.0)
        state_b = _make_state(timestamp=2.0)

        t1 = loop1.learn_step(state_a, "KITE", 5.0, state_b, False)
        t2 = loop2.learn_step(state_a, "KITE", 5.0, state_b, False)

        self.assertEqual(t1, t2)
        self.assertEqual(
            q1.get_q("combat", t1.state_key, "KITE"),
            q2.get_q("combat", t2.state_key, "KITE"),
        )

    def test_no_hardware_dependencies(self) -> None:
        """Ensures brain/learning_loop.py contains no imports of hardware/vision/control modules."""
        module_path = os.path.join(
            os.path.dirname(__file__), "..", "brain", "learning_loop.py"
        )
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()

        parsed = ast.parse(source)
        forbidden = {
            "control", "adb", "joystick", "opencv", "cv2", "yolo",
            "ultralytics", "realtime_vision", "ActionExecutor", "RewardCalculator",
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

    def test_integration_v2_regression(self) -> None:
        """Integration test: State A -> Action KITE -> reward +5 -> State B -> learn_step.

        Verifies Q-value changes according to V2 Temporal-Difference formula:
            Target = reward + gamma * max_a Q(s', a)
            Q(s, a) <- Q(s, a) + alpha * [Target - Q(s, a)]
        """
        loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.builder,
            q_learning=self.q_learning,
        )

        state_a = _make_state(timestamp=100.0, enemy_dist=200.0)  # Encodes to DANGER_CLOSE_HIGH_COMBO_True
        state_b = _make_state(timestamp=101.0, enemy_dist=350.0)  # Encodes to SWEET_SPOT_HIGH_COMBO_True
        action = Action(type=ActionType.KITE, direction=Vector2(-1.0, 0.0))

        # Setup initial Q-table condition:
        # Q(State A, kite) = 10.0
        # In State B: max future value Q(State B, sweet_spot_burst) = 20.0
        self.q_learning.set_q("combat", "DANGER_CLOSE_HIGH_COMBO_True", "kite", 10.0)
        self.q_learning.set_q("combat", "SWEET_SPOT_HIGH_COMBO_True", "sweet_spot_burst", 20.0)

        transition = loop.learn_step(
            state=state_a,
            action=action,
            reward=5.0,
            next_state=state_b,
            done=False,
        )

        # Expected calculation:
        # Target = 5.0 + 0.85 * 20.0 = 22.0
        # Update = 10.0 + 0.10 * (22.0 - 10.0) = 10.0 + 1.2 = 11.20
        self.assertEqual(transition.state_key, "DANGER_CLOSE_HIGH_COMBO_True")
        self.assertEqual(transition.action, "kite")
        self.assertEqual(transition.reward, 5.0)
        self.assertEqual(transition.next_state_key, "SWEET_SPOT_HIGH_COMBO_True")
        self.assertFalse(transition.done)

        new_q = self.q_learning.get_q("combat", "DANGER_CLOSE_HIGH_COMBO_True", "kite")
        self.assertAlmostEqual(new_q, 11.20, places=2)


if __name__ == "__main__":
    unittest.main()
