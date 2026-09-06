"""
tests/test_stage_21_runtime.py — Stage 21 V2 Runtime Orchestrator Tests.

Verifies:
A. One complete successful runtime tick.
B. Event propagation: WorldEventDetector -> LearningIntegrator.
C. Brain action reaches ActionExecutor.
D. No direct hardware calls from runtime.
E. Dependency injection with fake components.
F. Player temporary loss does not become player death.
G. PLAYER_DEATH produces terminal learning semantics (done=True, next_state=None).
H. TARGET_LOST with no reward produces no learning transitions.
I. Runtime shutdown/stop works.
J. A failed execution does not corrupt the learning pipeline.
K. Missing device/capture failure is reported clearly.
L. Runtime does not duplicate reward/Q-learning logic.
M. Runtime does not bypass ActionExecutor.
"""

import ast
import os
import sys
import unittest
from typing import List, Optional
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app.runtime import V2Runtime, RuntimeTickResult
from actions.executor import ActionExecutor, ExecutionResult
from actions.models import Action, ActionType
from brain.encoder import TacticalStateEncoder
from brain.learning_integration import LearningIntegrator
from brain.learning_loop import LearningLoop
from brain.q_learning import QLearningCore
from brain.rewards import RewardCalculator
from brain.state import TacticalState, TacticalStateBuilder
from brain.transition import Transition
from brain.transition_builder import TransitionBuilder
from vision.detector import Detection
from vision.hp_detector import HPObservation
from vision.skill_state import SkillState
from world.builder import WorldStateBuilder
from world.events import EventType, GameEvent, WorldEventDetector
from world.models import EnemyState, PlayerState, Vector2, WorldState
from world.tracker import Track, TrackLifecycle, WorldTracker


_SENTINEL = object()

class FakeCapture:
    def __init__(self, frame_to_return=_SENTINEL):
        if frame_to_return is _SENTINEL:
            self.frame = np.zeros((720, 1544, 3), dtype=np.uint8)
        else:
            self.frame = frame_to_return
        self.closed = False

    def grab(self):
        return self.frame

    def close(self):
        self.closed = True


class FakeDetector:
    def __init__(self, detections=None):
        self.detections = detections if detections is not None else []

    def detect(self, frame):
        return self.detections


class FakeBrain:
    def __init__(self, action_to_return=None):
        self.action_to_return = action_to_return or Action(type=ActionType.MOVE, direction=Vector2(900.0, 360.0))
        self.decide_calls = []

    def decide(self, state):
        self.decide_calls.append(state)
        return self.action_to_return


class FakeExecutor:
    def __init__(self, succeed=True):
        self.executed_actions = []
        self.succeed = succeed
        self.closed = False

    def execute(self, action: Action) -> ExecutionResult:
        self.executed_actions.append(action)
        if self.succeed:
            return ExecutionResult(success=True, action=action)
        return ExecutionResult(success=False, action=action, error="Simulated hardware failure")

    def close(self):
        self.closed = True


class FakeEventDetector:
    def __init__(self, events_to_return=None):
        self.events_to_return = events_to_return if events_to_return is not None else []
        self.update_calls = []

    def update(self, world_state, tracks, timestamp):
        self.update_calls.append((world_state, tracks, timestamp))
        return list(self.events_to_return)


class FakeLearningIntegrator:
    def __init__(self):
        self.process_calls = []

    def process(self, state, action, events, next_state, done):
        self.process_calls.append({
            "state": state,
            "action": action,
            "events": events,
            "next_state": next_state,
            "done": done,
        })
        # Simulate generating a transition
        action_name = action.type.value if isinstance(action, Action) else str(action)
        return [Transition(
            state_key="TEST_STATE",
            action=action_name,
            reward=10.0,
            next_state_key="NEXT_STATE" if next_state else None,
            done=done,
        )]


class TestStage21Runtime(unittest.TestCase):
    def setUp(self):
        self.current_time = 100.0
        self.time_provider = lambda: self.current_time

        self.capture = FakeCapture()
        self.detector = FakeDetector()
        self.world_builder = WorldStateBuilder()
        self.tracker = WorldTracker()
        self.event_detector = FakeEventDetector()
        self.tactical_state_builder = TacticalStateBuilder()
        self.brain = FakeBrain()
        self.executor = FakeExecutor(succeed=True)
        self.learning = FakeLearningIntegrator()

        self.runtime = V2Runtime(
            capture=self.capture,
            detector=self.detector,
            world_builder=self.world_builder,
            tracker=self.tracker,
            event_detector=self.event_detector,
            tactical_state_builder=self.tactical_state_builder,
            brain=self.brain,
            action_executor=self.executor,
            learning_integrator=self.learning,
            time_provider=self.time_provider,
        )

    # --------------------------------------------------------------------------
    # A. Complete successful runtime tick
    # --------------------------------------------------------------------------
    def test_a_complete_successful_runtime_tick(self):
        """A single runtime tick executes cleanly and produces a complete RuntimeTickResult."""
        res = self.runtime.tick()

        self.assertTrue(res.success)
        self.assertTrue(res.frame_captured)
        self.assertIsNotNone(res.world_state)
        self.assertIsNotNone(res.tactical_state)
        self.assertIsNotNone(res.action)
        self.assertIsNotNone(res.execution_result)
        self.assertTrue(res.execution_result.success)
        self.assertEqual(res.timestamp, 100.0)

    # --------------------------------------------------------------------------
    # B. Event propagation: WorldEventDetector -> LearningIntegrator
    # --------------------------------------------------------------------------
    def test_b_event_propagation_to_learning_integrator(self):
        """Events produced in step N are received by LearningIntegrator in step N+1."""
        # Tick 1: sets up initial action and state
        self.runtime.tick()

        # Inject event for tick 2
        hero_kill_event = GameEvent(
            type=EventType.HERO_KILL,
            timestamp=101.0,
            entity_id=1,
            confidence=1.0,
            evidence=("test_kill",),
        )
        self.event_detector.events_to_return = [hero_kill_event]
        self.current_time = 101.0

        # Tick 2: LearningIntegrator receives event and completes previous transition
        res2 = self.runtime.tick()

        self.assertEqual(len(self.learning.process_calls), 1)
        call = self.learning.process_calls[0]
        self.assertEqual(call["events"], [hero_kill_event])
        self.assertFalse(call["done"])
        self.assertIsNotNone(call["next_state"])

    # --------------------------------------------------------------------------
    # C. Brain action reaches ActionExecutor
    # --------------------------------------------------------------------------
    def test_c_brain_action_reaches_action_executor(self):
        """Brain decision is passed directly to ActionExecutor without bypass."""
        expected_action = Action(type=ActionType.CAST_S1)
        self.brain.action_to_return = expected_action

        res = self.runtime.tick()

        self.assertEqual(len(self.executor.executed_actions), 1)
        self.assertEqual(self.executor.executed_actions[0], expected_action)
        self.assertEqual(res.action, expected_action)

    # --------------------------------------------------------------------------
    # D. No direct hardware calls from runtime
    # --------------------------------------------------------------------------
    def test_d_no_direct_hardware_calls_from_runtime(self):
        """Runtime module does NOT import or instantiate ADBTransport or joystick directly."""
        runtime_path = os.path.join(WORKSPACE_ROOT, "app", "runtime.py")
        with open(runtime_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        forbidden = {"ADBTransport", "JoystickControl", "JoystickController", "subprocess"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], forbidden)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.assertNotIn(node.module.split(".")[0], forbidden)

    # --------------------------------------------------------------------------
    # E. Dependency injection with fake components
    # --------------------------------------------------------------------------
    def test_e_dependency_injection_with_custom_fakes(self):
        """Runtime accepts injected components and executes them without side effects."""
        custom_brain = FakeBrain(action_to_return=Action(type=ActionType.KITE, direction=Vector2(950.0, 360.0)))
        custom_executor = FakeExecutor()
        rt = V2Runtime(
            capture=self.capture,
            brain=custom_brain,
            action_executor=custom_executor,
            time_provider=self.time_provider,
        )
        res = rt.tick()
        self.assertTrue(res.success)
        self.assertEqual(len(custom_executor.executed_actions), 1)
        self.assertEqual(custom_executor.executed_actions[0].type, ActionType.KITE)

    # --------------------------------------------------------------------------
    # F. Player temporary loss != player death
    # --------------------------------------------------------------------------
    def test_f_player_temporary_loss_does_not_become_player_death(self):
        """LOST != DEAD: When player is not detected, is_player_dead remains False."""
        # No detections: player will be marked invisible
        res = self.runtime.tick()

        self.assertFalse(res.tactical_state.is_player_dead)
        self.assertFalse(res.tactical_state.is_player_visible)

    # --------------------------------------------------------------------------
    # G. PLAYER_DEATH produces terminal learning semantics
    # --------------------------------------------------------------------------
    def test_g_player_death_produces_terminal_learning_semantics(self):
        """PLAYER_DEATH event causes LearningIntegrator to process done=True with next_state=None."""
        self.runtime.tick()  # Tick 1

        # Tick 2: inject PLAYER_DEATH event
        death_event = GameEvent(
            type=EventType.PLAYER_DEATH,
            timestamp=101.0,
            entity_id=None,
            confidence=1.0,
            evidence=("died",),
        )
        self.event_detector.events_to_return = [death_event]
        self.current_time = 101.0

        res2 = self.runtime.tick()

        self.assertEqual(len(self.learning.process_calls), 1)
        call = self.learning.process_calls[0]
        self.assertTrue(call["done"])
        self.assertIsNone(call["next_state"])

    # --------------------------------------------------------------------------
    # H. TARGET_LOST produces no learning transitions
    # --------------------------------------------------------------------------
    def test_h_target_lost_with_no_reward_produces_no_transitions(self):
        """Real LearningIntegrator produces 0 transitions for TARGET_LOST."""
        # Use real LearningIntegrator components
        encoder = TacticalStateEncoder()
        trans_builder = TransitionBuilder(encoder)
        q_learn = QLearningCore()
        rewards = RewardCalculator()
        loop = LearningLoop(encoder, trans_builder, q_learn)
        real_integrator = LearningIntegrator(rewards, trans_builder, loop)

        rt = V2Runtime(
            capture=self.capture,
            event_detector=self.event_detector,
            learning_integrator=real_integrator,
            time_provider=self.time_provider,
        )

        rt.tick()  # Tick 1

        lost_event = GameEvent(
            type=EventType.TARGET_LOST,
            timestamp=101.0,
            entity_id=5,
            confidence=1.0,
            evidence=("lost",),
        )
        self.event_detector.events_to_return = [lost_event]
        self.current_time = 101.0

        res2 = rt.tick()
        # No transitions generated for TARGET_LOST
        self.assertEqual(len(res2.transitions), 0)

    # --------------------------------------------------------------------------
    # I. Runtime shutdown / stop works
    # --------------------------------------------------------------------------
    def test_i_runtime_shutdown_and_stop(self):
        """run_loop terminates properly when max_ticks reached and invokes stop()."""
        ticks = self.runtime.run_loop(max_ticks=3)
        self.assertEqual(ticks, 3)
        self.assertFalse(self.runtime.is_running)
        self.assertTrue(self.capture.closed)
        self.assertTrue(self.executor.closed)

    # --------------------------------------------------------------------------
    # J. Failed execution does not corrupt learning pipeline
    # --------------------------------------------------------------------------
    def test_j_failed_execution_does_not_corrupt_learning_pipeline(self):
        """When action execution fails, learning does not process a corrupt transition."""
        self.executor.succeed = False  # Make execution fail
        self.runtime.tick()  # Tick 1 fails execution

        # Tick 2 has an event
        self.event_detector.events_to_return = [
            GameEvent(EventType.HERO_KILL, 101.0, 1, 1.0, ("kill",))
        ]
        self.current_time = 101.0
        self.runtime.tick()

        # Learning process should NOT be called because previous execution failed
        self.assertEqual(len(self.learning.process_calls), 0)

    # --------------------------------------------------------------------------
    # K. Missing device / capture failure reported clearly
    # --------------------------------------------------------------------------
    def test_k_missing_capture_reported_clearly(self):
        """When capture returns None (e.g. Scrcpy not running), tick returns clean error."""
        broken_capture = FakeCapture(frame_to_return=None)
        rt = V2Runtime(
            capture=broken_capture,
            time_provider=self.time_provider,
        )
        res = rt.tick()
        self.assertFalse(res.success)
        self.assertFalse(res.frame_captured)
        self.assertIn("Capture returned None", res.error)

    # --------------------------------------------------------------------------
    # L. Runtime does not duplicate reward / Q-learning logic
    # --------------------------------------------------------------------------
    def test_l_runtime_does_not_duplicate_learning_logic(self):
        """Runtime contains no Q-tables, reward constants, or math updates."""
        runtime_path = os.path.join(WORKSPACE_ROOT, "app", "runtime.py")
        with open(runtime_path, "r", encoding="utf-8") as f:
            content = f.read()

        # No reward constants
        self.assertNotIn("KILL_REWARD", content)
        self.assertNotIn("DEATH_PENALTY", content)
        # No Q-learning formulas
        self.assertNotIn("learning_rate", content)
        self.assertNotIn("discount_factor", content)

    # --------------------------------------------------------------------------
    # M. Runtime does not bypass ActionExecutor
    # --------------------------------------------------------------------------
    def test_m_runtime_does_not_bypass_action_executor(self):
        """Actions decided by Brain are strictly executed via ActionExecutor."""
        rt = V2Runtime(
            capture=self.capture,
            brain=self.brain,
            action_executor=self.executor,
            time_provider=self.time_provider,
        )
        rt.tick()
        self.assertEqual(len(self.executor.executed_actions), 1)


if __name__ == "__main__":
    unittest.main()
