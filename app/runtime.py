"""
app.runtime — V2 Runtime Orchestrator for Claude Tactical AI.

Connects the validated modular subsystems into an end-to-end frame/tick execution loop:
    Capture
       ↓
     Vision
       ↓
   WorldState
       ↓
    Tracker
       ↓
     Events  ──────────────┐
       ↓                   ↓
  TacticalState    LearningIntegrator
       ↓                   ↓
     Brain            LearningLoop
       ↓                   ↓
     Action           QLearningCore
       ↓
 ActionExecutor
       ↓
    Control
       ↓
   ADBTransport
       ↓
    Android

Zero monolithic logic: pure composition and orchestration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Union

from actions.executor import ActionExecutor, ExecutionResult
from actions.models import Action, ActionType
from brain.learning_integration import LearningIntegrator
from brain.state import TacticalState, TacticalStateBuilder
from brain.transition import Transition
from world.builder import WorldStateBuilder
from world.events import EventType, GameEvent, WorldEventDetector
from world.models import WorldState
from world.tracker import Track, WorldTracker

logger = logging.getLogger("V2Runtime")


@dataclass(frozen=True)
class RuntimeTickResult:
    """
    Structured outcome of a single runtime frame/tick execution.
    """
    success: bool
    timestamp: float
    frame_captured: bool
    world_state: Optional[WorldState] = None
    tactical_state: Optional[TacticalState] = None
    events: List[GameEvent] = field(default_factory=list)
    action: Optional[Action] = None
    execution_result: Optional[ExecutionResult] = None
    transitions: List[Transition] = field(default_factory=list)
    error: Optional[str] = None


class V2Runtime:
    """
    V2 Runtime Orchestrator.
    Orchestrates frame capture, computer vision, world tracking, event detection,
    tactical brain decision, action execution, and RL learning updates.
    """

    def __init__(
        self,
        capture: Any,
        detector: Optional[Any] = None,
        world_builder: Optional[WorldStateBuilder] = None,
        tracker: Optional[WorldTracker] = None,
        event_detector: Optional[WorldEventDetector] = None,
        tactical_state_builder: Optional[TacticalStateBuilder] = None,
        brain: Optional[Any] = None,
        action_executor: Optional[ActionExecutor] = None,
        learning_integrator: Optional[LearningIntegrator] = None,
        hp_detector: Optional[Any] = None,
        skill_checker: Optional[Any] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        """
        Initializes V2Runtime with dependency-injected components.
        """
        self.capture = capture
        self.detector = detector
        self.world_builder = world_builder or WorldStateBuilder()
        self.tracker = tracker or WorldTracker()
        self.event_detector = event_detector or WorldEventDetector()
        self.tactical_state_builder = tactical_state_builder or TacticalStateBuilder()
        self.brain = brain
        self.action_executor = action_executor
        self.learning_integrator = learning_integrator
        self.hp_detector = hp_detector
        self.skill_checker = skill_checker
        self.time_provider = time_provider or time.time

        self._running: bool = False
        self._prev_tactical_state: Optional[TacticalState] = None
        self._prev_action: Optional[Action] = None
        self._prev_execution_success: bool = True
        self.total_ticks: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def tick(self, timestamp: Optional[float] = None) -> RuntimeTickResult:
        """
        Executes one full cycle of the V2 runtime pipeline.

        Returns:
            RuntimeTickResult describing the execution diagnostics.
        """
        now = self.time_provider() if timestamp is None else float(timestamp)
        self.total_ticks += 1

        # ----------------------------------------------------------------------
        # 1. Screen Capture
        # ----------------------------------------------------------------------
        frame = None
        if self.capture:
            if hasattr(self.capture, "grab"):
                frame = self.capture.grab()
            elif hasattr(self.capture, "capture"):
                frame = self.capture.capture()
            elif callable(self.capture):
                frame = self.capture()

        if frame is None:
            return RuntimeTickResult(
                success=False,
                timestamp=now,
                frame_captured=False,
                error="Capture returned None (device or scrcpy window unavailable)",
            )

        try:
            # ------------------------------------------------------------------
            # 2. Vision & Detections
            # ------------------------------------------------------------------
            detections = []
            if self.detector:
                if hasattr(self.detector, "detect"):
                    detections = self.detector.detect(frame)
                elif callable(self.detector):
                    detections = self.detector(frame)

            # ------------------------------------------------------------------
            # 3. World Tracking
            # ------------------------------------------------------------------
            tracks: List[Track] = []
            if self.tracker and hasattr(self.tracker, "update"):
                tracks = self.tracker.update(detections, now)

            # ------------------------------------------------------------------
            # 4. Sensor Readings (Optical HP & Skill State)
            # ------------------------------------------------------------------
            player_hp = None
            if self.hp_detector and hasattr(self.hp_detector, "detect"):
                player_hp = self.hp_detector.detect(frame)

            skills = None
            if self.skill_checker and hasattr(self.skill_checker, "check"):
                skills = self.skill_checker.check(frame)

            # ------------------------------------------------------------------
            # 5. WorldState Construction
            # ------------------------------------------------------------------
            world_state = self.world_builder.build(
                timestamp=now,
                detections=detections,
                tracks=tracks,
                player_hp=player_hp,
                skills=skills,
            )

            # ------------------------------------------------------------------
            # 6. Event Detection
            # ------------------------------------------------------------------
            events: List[GameEvent] = []
            if self.event_detector and hasattr(self.event_detector, "update"):
                events = self.event_detector.update(
                    world_state=world_state,
                    tracks=tracks,
                    timestamp=now,
                )

            # ------------------------------------------------------------------
            # 7. TacticalState Feature Extraction
            # ------------------------------------------------------------------
            tactical_state = self.tactical_state_builder.build(world_state)

            # ------------------------------------------------------------------
            # 8. Reinforcement Learning Side-Channel
            # ------------------------------------------------------------------
            transitions: List[Transition] = []
            if (
                self.learning_integrator
                and self._prev_tactical_state is not None
                and self._prev_action is not None
            ):
                # A failed physical execution does NOT corrupt the learning pipeline
                if self._prev_execution_success:
                    is_terminal = (
                        any(e.type == EventType.PLAYER_DEATH for e in events)
                        or tactical_state.is_player_dead
                    )
                    next_state = None if is_terminal else tactical_state

                    transitions = self.learning_integrator.process(
                        state=self._prev_tactical_state,
                        action=self._prev_action,
                        events=events,
                        next_state=next_state,
                        done=is_terminal,
                    )

            # ------------------------------------------------------------------
            # 9. Tactical Brain Decision
            # ------------------------------------------------------------------
            action = None
            if self.brain and hasattr(self.brain, "decide"):
                # Support both TacticalState contract (ClaudeBrainV2)
                # and WorldState contract (TacticalBrain / V1BrainAdapter)
                try:
                    action = self.brain.decide(tactical_state)
                except TypeError:
                    action = self.brain.decide(world_state)

            if action is None:
                action = Action(type=ActionType.IDLE)

            # ------------------------------------------------------------------
            # 10. Action Execution (via ActionExecutor)
            # ------------------------------------------------------------------
            execution_result = None
            if self.action_executor and hasattr(self.action_executor, "execute"):
                execution_result = self.action_executor.execute(action)
            else:
                execution_result = ExecutionResult(success=True, action=action)

            # Record state for the next tick's transition
            self._prev_tactical_state = tactical_state
            self._prev_action = action
            self._prev_execution_success = bool(execution_result.success)

            return RuntimeTickResult(
                success=True,
                timestamp=now,
                frame_captured=True,
                world_state=world_state,
                tactical_state=tactical_state,
                events=events,
                action=action,
                execution_result=execution_result,
                transitions=transitions,
            )

        except Exception as e:
            logger.exception("Error executing runtime tick: %s", e)
            return RuntimeTickResult(
                success=False,
                timestamp=now,
                frame_captured=True,
                error=str(e),
            )

    def run_loop(
        self,
        max_ticks: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> int:
        """
        Runs the runtime tick loop until stopped or max_ticks reached.

        Args:
            max_ticks: Optional limit on total executed frames.
            target_fps: Optional target FPS frame limiter.

        Returns:
            Number of ticks executed in this run.
        """
        self._running = True
        ticks_executed = 0
        target_interval = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0

        try:
            while self._running:
                t0 = time.perf_counter()
                self.tick()
                ticks_executed += 1

                if max_ticks is not None and ticks_executed >= max_ticks:
                    break

                if target_interval > 0.0:
                    elapsed = time.perf_counter() - t0
                    sleep_time = target_interval - elapsed
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)
        finally:
            self.stop()

        return ticks_executed

    def stop(self) -> None:
        """Stops the runtime loop and performs clean component shutdown."""
        self._running = False
        if self.capture and hasattr(self.capture, "close"):
            try:
                self.capture.close()
            except Exception:
                pass

        if self.action_executor and hasattr(self.action_executor, "close"):
            try:
                self.action_executor.close()
            except Exception:
                pass

    def reset_state(self) -> None:
        """Clears previous frame history."""
        self._prev_tactical_state = None
        self._prev_action = None
        self._prev_execution_success = True
