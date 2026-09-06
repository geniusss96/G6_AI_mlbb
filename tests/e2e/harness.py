"""
tests.e2e.harness — Deterministic offline End-to-End test harness for Claude Tactical AI V2.

Assembles the full V2 pipeline:
Capture -> Vision -> WorldState -> Tracker -> Events -> TacticalState -> Brain -> Action -> ActionExecutor -> Control -> ADB
and the learning side-channel:
Events -> RewardCalculator -> LearningIntegrator -> LearningLoop -> QLearningCore -> Memory.

Zero dependency on real Android, ADB, Scrcpy, GPU, physical screen, or time.sleep.
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
import numpy as np

from app.runtime import V2Runtime, RuntimeTickResult
from actions.executor import ActionExecutor
from actions.models import Action, ActionType
from brain.brain import ClaudeBrainV2
from brain.encoder import TacticalStateEncoder
from brain.learning_integration import LearningIntegrator
from brain.learning_loop import LearningLoop
from brain.memory import BrainMemory
from brain.q_learning import QLearningCore
from brain.rewards import RewardCalculator
from brain.state import TacticalStateBuilder
from brain.transition_builder import TransitionBuilder
from control.attack import AttackControl
from control.humanizer import InputHumanizer
from control.joystick import JoystickControl
from control.skills import SkillsControl
from vision.detector import Detection
from vision.hp_detector import HPObservation
from vision.skill_state import SkillState
from world.builder import WorldStateBuilder
from world.events import WorldEventDetector, GameEvent
from world.models import Vector2, WorldState
from world.tracker import WorldTracker


class FakeADBTransport:
    """Mock ADBTransport that captures commands in memory without physical hardware."""

    def __init__(self):
        self.sent_commands: List[str] = []
        self.taps: List[Tuple[int, int]] = []
        self.swipes: List[Tuple[int, int, int, int, int]] = []
        self.motionevents: List[Tuple[str, int, int]] = []
        self.is_connected: bool = True

    def send(self, command: str) -> bool:
        if not self.is_connected:
            return False
        self.sent_commands.append(command)
        return True

    def send_tap(self, x: int, y: int) -> bool:
        if not self.is_connected:
            return False
        self.taps.append((x, y))
        self.sent_commands.append(f"input tap {x} {y}")
        return True

    def send_swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 100) -> bool:
        if not self.is_connected:
            return False
        self.swipes.append((x1, y1, x2, y2, duration_ms))
        self.sent_commands.append(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
        return True

    def send_motionevent(self, action: str, x: int, y: int) -> bool:
        if not self.is_connected:
            return False
        self.motionevents.append((action, x, y))
        self.sent_commands.append(f"input motionevent {action} {x} {y}")
        return True

    def is_alive(self) -> bool:
        return self.is_connected

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class SimulatedCapture:
    """Simulates frame acquisition with optional simulated hardware failure."""

    def __init__(self, width: int = 1544, height: int = 720):
        self.width = width
        self.height = height
        self.synthetic_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.should_fail: bool = False
        self.closed: bool = False

    def grab(self) -> Optional[np.ndarray]:
        if self.should_fail or self.closed:
            return None
        return self.synthetic_frame

    def close(self) -> None:
        self.closed = True


class SimulatedVision:
    """Configurable vision detector returning synthetic battlefield entity detections."""

    def __init__(self):
        self.next_detections: List[Detection] = []

    def set_detections(self, detections: List[Detection]) -> None:
        self.next_detections = list(detections)

    def clear(self) -> None:
        self.next_detections.clear()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return list(self.next_detections)


class SimulatedControl:
    """Bundles V2 control implementations backed by FakeADBTransport."""

    def __init__(self, time_provider: Callable[[], float]):
        self.transport = FakeADBTransport()
        self.humanizer = InputHumanizer()
        self.joystick = JoystickControl(
            transport=self.transport,  # type: ignore
            time_provider=time_provider,
        )
        self.attack = AttackControl(
            transport=self.transport,  # type: ignore
            humanizer=self.humanizer,
            joystick=self.joystick,
            time_provider=time_provider,
        )
        self.skills = SkillsControl(
            transport=self.transport,  # type: ignore
            humanizer=self.humanizer,
            joystick=self.joystick,
        )
        self.executor = ActionExecutor(
            joystick=self.joystick,
            skills=self.skills,
            attack_controller=self.attack,
            hero_pos_provider=lambda: (772.0, 360.0),
        )


class E2EHarness:
    """
    Complete End-to-End deterministic simulation harness for V2 architecture.
    """

    def __init__(self, initial_time: float = 100.0):
        self.current_time = float(initial_time)

        # 1. Capture & Vision
        self.capture = SimulatedCapture()
        self.vision = SimulatedVision()

        # 2. World layer
        self.world_builder = WorldStateBuilder()
        self.tracker = WorldTracker()
        self.event_detector = WorldEventDetector()

        # 3. Tactical State layer
        self.tactical_state_builder = TacticalStateBuilder()

        # 4. Cognitive & Learning layer
        self.q_learning = QLearningCore(learning_rate=0.25, discount_factor=0.85)
        self.rewards = RewardCalculator()
        self.memory = BrainMemory()
        self.encoder = TacticalStateEncoder()
        self.transition_builder = TransitionBuilder(self.encoder)
        self.learning_loop = LearningLoop(
            encoder=self.encoder,
            transition_builder=self.transition_builder,
            q_learning=self.q_learning,
            memory=self.memory,
        )
        self.learning_integrator = LearningIntegrator(
            reward_calculator=self.rewards,
            transition_builder=self.transition_builder,
            learning_loop=self.learning_loop,
        )
        self.brain = ClaudeBrainV2(
            q_learning=self.q_learning,
            rewards=self.rewards,
            memory=self.memory,
        )

        # 5. Control layer
        self.control = SimulatedControl(time_provider=self.get_time)

        # 6. Runtime Orchestrator
        self.runtime = V2Runtime(
            capture=self.capture,
            detector=self.vision,
            world_builder=self.world_builder,
            tracker=self.tracker,
            event_detector=self.event_detector,
            tactical_state_builder=self.tactical_state_builder,
            brain=self.brain,
            action_executor=self.control.executor,
            learning_integrator=self.learning_integrator,
            time_provider=self.get_time,
        )

    def get_time(self) -> float:
        return self.current_time

    def advance_time(self, seconds: float) -> float:
        self.current_time += float(seconds)
        return self.current_time

    def tick(self) -> RuntimeTickResult:
        """Executes one runtime tick at current simulated time."""
        return self.runtime.tick(timestamp=self.current_time)

    def spawn_enemy_hero(self, x: float = 900.0, y: float = 360.0, conf: float = 0.95) -> Detection:
        det = Detection(
            class_id=0,
            class_name="enemy_hero",
            confidence=conf,
            x1=x - 20,
            y1=y - 20,
            x2=x + 20,
            y2=y + 20,
        )
        self.vision.set_detections([det])
        return det

    def spawn_minion(self, x: float = 850.0, y: float = 360.0, conf: float = 0.85) -> Detection:
        det = Detection(
            class_id=1,
            class_name="minion",
            confidence=conf,
            x1=x - 15,
            y1=y - 15,
            x2=x + 15,
            y2=y + 15,
        )
        self.vision.set_detections([det])
        return det

    def clear_entities(self) -> None:
        self.vision.clear()
