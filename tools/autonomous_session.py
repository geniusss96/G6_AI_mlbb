"""
tools/autonomous_session.py -- Stage 29.1: Long Isolated Autonomous Gameplay Session.

Runs a bounded autonomous MLBB gameplay session using the complete V2 pipeline:
    REAL SCREEN → VISION → WORLD STATE → TACTICAL BRAIN → ACTION →
    ACTION EXECUTOR → CONTROL → ANDROID → OBSERVED EVENT → REWARD →
    TRANSITION → ISOLATED Q UPDATE

Uses ALL existing V2 components. Creates NO new runtime, NO duplicated logic.

Safety:
    - Isolated QLearningCore (writes only to data/q_brain_isolated_stage29.json)
    - Production q_brain.json and q_brain_baseline.json are NEVER modified
    - Bounded session (10 minutes / 10,000 ticks max)
    - Periodic isolated-Q checkpoints so a long run is not lost on interruption
    - Behavioral watchdogs (same-action, one-action collapse, event storm)
    - Clean emergency shutdown (joystick release, ADB close, Scrcpy terminate)

Usage:
    1. Manually launch MLBB and enter practice/training match on Android device.
    2. python tools/autonomous_session.py
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import statistics
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_RUNTIME_SECONDS = 600
MAX_TICKS = 10000
MAX_CONSECUTIVE_FAILURES = 5
OBSERVE_TICKS = 8          # observe-only before enabling actions
SAME_ACTION_WARN = 10      # consecutive identical actions → warning
SAME_ACTION_STOP = 20      # consecutive identical actions → stop
ONE_ACTION_DOMINANCE = 0.80 # >80% same type → flag
EVENT_STORM_THRESHOLD = 10  # eventful ticks in 1 second → stop
EVENT_STORM_WINDOW = 1.0

Q_BRAIN_PATH = os.path.join(WORKSPACE_ROOT, "q_brain.json")
BASELINE_PATH = os.path.join(WORKSPACE_ROOT, "data", "q_brain_baseline.json")
ISOLATED_Q_PATH = os.path.join(WORKSPACE_ROOT, "data", "q_brain_isolated_stage29.json")
SESSION_LOG_PATH = os.path.join(WORKSPACE_ROOT, "data", "stage29_session_log.jsonl")
SUMMARY_PATH = os.path.join(WORKSPACE_ROOT, "data", "stage29_summary.json")
CHECKPOINT_INTERVAL_SECONDS = 60.0

SEP = "=" * 72

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AutonomousSession")


class RuntimeMode(Enum):
    OBSERVE = "OBSERVE"
    AUTONOMOUS_ISOLATED = "AUTONOMOUS_ISOLATED"


class StopReason(Enum):
    TIME_LIMIT = "TIME_LIMIT"
    TICK_LIMIT = "TICK_LIMIT"
    FAILURE_LIMIT = "FAILURE_LIMIT"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    WATCHDOG_SAME_ACTION = "WATCHDOG_SAME_ACTION"
    WATCHDOG_EVENT_STORM = "WATCHDOG_EVENT_STORM"
    USER_INTERRUPT = "USER_INTERRUPT"
    OBSERVE_FAILED = "OBSERVE_FAILED"


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pct(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] * (1.0 - (k - lo)) + s[hi] * (k - lo)


# ---------------------------------------------------------------------------
# Per-decision compact log record
# ---------------------------------------------------------------------------
@dataclass
class DecisionRecord:
    tick: int
    timestamp: float
    mode: str
    state_summary: str = ""
    action_type: str = ""
    action_direction: Optional[str] = None
    action_target: Optional[int] = None
    execution_ok: bool = True
    execution_error: Optional[str] = None
    events: List[str] = field(default_factory=list)
    rewards: List[Dict[str, Any]] = field(default_factory=list)
    transitions: int = 0
    q_updates: int = 0


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------
@dataclass
class SessionStats:
    mode: str = ""
    duration_sec: float = 0.0
    ticks: int = 0
    avg_tick_ms: float = 0.0
    p95_tick_ms: float = 0.0
    stop_reason: str = ""
    # Actions
    actions_total: int = 0
    actions_successful: int = 0
    actions_failed: int = 0
    action_distribution: Dict[str, int] = field(default_factory=dict)
    repetition_rate: float = 0.0
    dominant_action: Optional[str] = None
    # Events
    events_hero_kill: int = 0
    events_creep_kill: int = 0
    events_player_death: int = 0
    events_target_entered: int = 0
    events_target_lost: int = 0
    events_total: int = 0
    # Rewards
    rewards_count: int = 0
    rewards_combat_total: float = 0.0
    rewards_farm_total: float = 0.0
    rewards_roam_total: float = 0.0
    rewards_positive_total: float = 0.0
    rewards_negative_total: float = 0.0
    # Learning
    transitions_total: int = 0
    q_updates_total: int = 0
    q_states_before: int = 0
    q_states_after: int = 0
    isolated_q_path: str = ""
    # Safety
    watchdog_warnings: List[str] = field(default_factory=list)
    watchdog_stops: int = 0
    emergency_stop: bool = False
    joystick_released: bool = False
    teardown_ok: bool = False
    # Hashes
    q_brain_sha_before: str = ""
    q_brain_sha_after: str = ""
    baseline_sha_before: str = ""
    baseline_sha_after: str = ""
    isolated_q_sha: str = ""


def count_q_states(ql) -> int:
    """Count unique states across all Q-tables."""
    total = 0
    for table_name in ("combat", "farm", "roam"):
        table = ql.get_table(table_name)
        total += len(table)
    return total


# ---------------------------------------------------------------------------
# Durable isolated-Q checkpoint
# ---------------------------------------------------------------------------
def save_isolated_q(ql, path: str, metadata: Dict[str, Any]) -> str:
    """Atomically persist isolated Q-learning state and return its SHA-256."""
    data = ql.export_tables()
    data["metadata"] = metadata
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return sha256(path)


def save_session_log(records: List[DecisionRecord], path: str) -> None:
    """Atomically write the current compact session log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(f"\n{SEP}")
    print("  STAGE 29.1 -- LONG ISOLATED AUTONOMOUS TRAINING SESSION")
    print(SEP)

    stats = SessionStats(mode=RuntimeMode.AUTONOMOUS_ISOLATED.value)
    scrcpy_proc = None
    joystick = None
    adb = None
    session_log: List[DecisionRecord] = []

    # ===================================================================
    # 0. Pre-flight: Q-memory hashes
    # ===================================================================
    stats.q_brain_sha_before = sha256(Q_BRAIN_PATH)
    stats.baseline_sha_before = sha256(BASELINE_PATH)
    print(f"\n  PRE q_brain.json SHA-256:     {stats.q_brain_sha_before[:24]}...")
    print(f"  PRE q_brain_baseline SHA-256: {stats.baseline_sha_before[:24]}...")

    try:
        # ===================================================================
        # 1. Launch Scrcpy
        # ===================================================================
        from config.config import (
            PROJECT_ROOT, SCRCPY_PATH, SCRCPY_DIR, ADB_PATH,
            DEVICE_SERIAL, WEIGHTS_PATH, YOLO_IMGSZ, YOLO_DEVICE,
            YOLO_CUDA_STRICT,
        )

        print(f"\n[1] Preparing Scrcpy...")
        print(f"  YOLO_DEVICE = {YOLO_DEVICE}")
        print(f"  YOLO_CUDA_STRICT = {YOLO_CUDA_STRICT}")

        # Reuse an already-running Scrcpy window so the autonomous session
        # can coexist with a manually launched Scrcpy/Scrcpy session.
        from vision.capture import ScreenCapture
        capture = ScreenCapture()
        hwnd = capture.find_scrcpy_window()

        if hwnd:
            print(f"  Existing Scrcpy detected, reusing HWND={hwnd}")
            # scrcpy_proc stays None, so teardown will NOT close the
            # user-owned/manual Scrcpy process.
        else:
            print("  No existing Scrcpy found, launching bundled Scrcpy...")
            scrcpy_proc = subprocess.Popen(
                [SCRCPY_PATH, "-s", DEVICE_SERIAL, "--window-title", "scrcpy"],
                cwd=SCRCPY_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            t_wait = time.time()
            texture_ready = False
            while time.time() - t_wait < 10.0:
                line = scrcpy_proc.stdout.readline()
                if "Texture:" in line:
                    texture_ready = True
                    print(f"  [scrcpy] {line.strip()}")
                    break
                if line.strip():
                    print(f"  [scrcpy] {line.strip()}")
                if scrcpy_proc.poll() is not None:
                    break

            if not texture_ready:
                print("FATAL: Scrcpy texture not ready.")
                stats.stop_reason = StopReason.RUNTIME_ERROR.value
                return stats

            time.sleep(0.5)
            hwnd = capture.find_scrcpy_window()
            if not hwnd:
                print("FATAL: Could not find Scrcpy window.")
                stats.stop_reason = StopReason.RUNTIME_ERROR.value
                return stats
            print(f"  HWND={hwnd}")

        first_frame = capture.grab(hwnd=hwnd)
        if first_frame is None or first_frame.shape != (720, 1544, 3):
            print(f"FATAL: Bad first frame: {first_frame.shape if first_frame is not None else None}")
            stats.stop_reason = StopReason.RUNTIME_ERROR.value
            return stats
        print(f"  First frame shape: {first_frame.shape}")

        # ===================================================================
        # 2. Build V2 components
        # ===================================================================
        print(f"\n[2] Building V2 components...")

        from vision.detector import YoloDetector
        from vision.hp_detector import HPDetector
        from vision.skill_state import SkillStateChecker
        from world.tracker import WorldTracker
        from world.builder import WorldStateBuilder
        from world.events import WorldEventDetector, EventType
        from brain.state import TacticalStateBuilder
        from brain.q_learning import QLearningCore
        from brain.memory import BrainMemory
        from brain.brain import ClaudeBrainV2
        from brain.encoder import TacticalStateEncoder
        from brain.transition_builder import TransitionBuilder
        from brain.learning_loop import LearningLoop
        from brain.learning_integration import LearningIntegrator
        from brain.rewards import RewardCalculator
        from control.adb import ADBTransport
        from control.humanizer import InputHumanizer
        from control.joystick import JoystickControl
        from control.attack import AttackControl
        from control.skills import SkillsControl
        from actions.executor import ActionExecutor
        from actions.models import Action, ActionType

        # YOLO detector (production CUDA)
        print(f"  Building YoloDetector(device={YOLO_DEVICE!r}, cuda_strict={YOLO_CUDA_STRICT})...")
        detector = YoloDetector(
            weights_path=WEIGHTS_PATH,
            device=YOLO_DEVICE,
            imgsz=YOLO_IMGSZ,
            cuda_strict=YOLO_CUDA_STRICT,
        )

        # Cold inference warmup
        print("  Cold inference (JIT warmup)...")
        t0 = time.perf_counter()
        _ = detector.detect(first_frame)
        cold_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  Cold inference: {cold_ms:.1f} ms")

        # Sensors
        hp_raw = HPDetector()
        class _HP:
            def detect(self, f):
                return hp_raw.detect_player_hp(f, (772.0, 360.0))
        hp_adapter = _HP()
        skill_checker = SkillStateChecker()

        # World layer
        tracker = WorldTracker()
        world_builder = WorldStateBuilder()
        event_detector = WorldEventDetector()
        tactical_builder = TacticalStateBuilder()

        # ISOLATED learning pipeline (production Q NEVER touched).
        # Resume the existing Stage 29 isolated memory when it is valid; this
        # lets repeated training runs accumulate experience without ever
        # loading or mutating production q_brain.json.
        isolated_tables = {"roam_q": None, "farm_q": None, "combat_q": None}
        if os.path.exists(ISOLATED_Q_PATH):
            try:
                with open(ISOLATED_Q_PATH, "r", encoding="utf-8") as f:
                    saved_q = json.load(f)
                for key in isolated_tables:
                    value = saved_q.get(key)
                    isolated_tables[key] = value if isinstance(value, dict) else None
                print(f"  Resuming isolated Q: {ISOLATED_Q_PATH}")
            except Exception as e:
                logger.warning(f"Could not load isolated Q; starting fresh: {e}")

        isolated_ql = QLearningCore(
            roam_q=isolated_tables["roam_q"],
            farm_q=isolated_tables["farm_q"],
            combat_q=isolated_tables["combat_q"],
        )
        isolated_memory = BrainMemory()
        encoder = TacticalStateEncoder()
        transition_builder = TransitionBuilder(encoder=encoder)
        learning_loop = LearningLoop(
            encoder=encoder,
            transition_builder=transition_builder,
            q_learning=isolated_ql,
            memory=isolated_memory,
        )
        reward_calculator = RewardCalculator()
        learning_integrator = LearningIntegrator(
            reward_calculator=reward_calculator,
            transition_builder=transition_builder,
            learning_loop=learning_loop,
        )

        stats.q_states_before = count_q_states(isolated_ql)

        # Brain (uses isolated Q for policy evaluation too)
        brain = ClaudeBrainV2(
            q_learning=isolated_ql,
            memory=isolated_memory,
        )

        # Control layer
        adb = ADBTransport(binary=ADB_PATH, device_serial=DEVICE_SERIAL, auto_start=True)
        print(f"  ADB connected: {adb.is_alive()}")
        if not adb.is_alive():
            print("FATAL: ADB transport not alive.")
            stats.stop_reason = StopReason.RUNTIME_ERROR.value
            return stats

        humanizer = InputHumanizer()
        joystick = JoystickControl(transport=adb, time_provider=time.time)
        attack_ctrl = AttackControl(transport=adb, humanizer=humanizer,
                                    joystick=joystick, time_provider=time.time)
        skills_ctrl = SkillsControl(transport=adb, humanizer=humanizer, joystick=joystick)
        action_executor = ActionExecutor(
            joystick=joystick,
            skills=skills_ctrl,
            attack_controller=attack_ctrl,
            hero_pos_provider=lambda: (772.0, 360.0),
        )

        # Build V2Runtime for the tick method
        from app.runtime import V2Runtime
        runtime = V2Runtime(
            capture=capture,
            detector=detector,
            world_builder=world_builder,
            tracker=tracker,
            event_detector=event_detector,
            tactical_state_builder=tactical_builder,
            brain=brain,
            action_executor=None,  # Start with OBSERVE mode (no physical actions)
            learning_integrator=None,  # No learning in OBSERVE mode
            hp_detector=hp_adapter,
            skill_checker=skill_checker,
        )

        print(f"\n  All V2 components built successfully.")

        # ===================================================================
        # 3. OBSERVE phase (validate live screen before enabling actions)
        # ===================================================================
        print(f"\n{SEP}")
        print(f"  PHASE 1: OBSERVE ({OBSERVE_TICKS} ticks, no physical actions)")
        print(SEP)

        observe_ok = True
        for i in range(OBSERVE_TICKS):
            result = runtime.tick()
            if not result.success:
                print(f"  OBSERVE tick {i+1}: FAILED - {result.error}")
                if i >= 2:  # Allow first 1-2 to fail (initial frame sync)
                    observe_ok = False
                continue

            action_str = result.action.type.value if result.action else "None"
            det_count = (len(result.world_state.enemies) + len(result.world_state.minions) + len(result.world_state.turrets)) if result.world_state else 0
            event_count = len(result.events)
            tactical = result.tactical_state
            state_desc = "?"
            if tactical:
                if tactical.enemy_visible:
                    state_desc = f"COMBAT d={tactical.nearest_enemy_distance:.0f}" if tactical.nearest_enemy_distance else "COMBAT"
                elif tactical.minion_count > 0:
                    state_desc = f"FARM m={tactical.minion_count}"
                else:
                    state_desc = "ROAM"

            print(f"  OBSERVE tick {i+1}: dets={det_count} state={state_desc} "
                  f"brain={action_str} events={event_count} [OK]")

            rec = DecisionRecord(
                tick=i + 1,
                timestamp=result.timestamp,
                mode=RuntimeMode.OBSERVE.value,
                state_summary=state_desc,
                action_type=action_str,
                events=[e.type.value for e in result.events],
            )
            session_log.append(rec)

        if not observe_ok:
            print("\n  OBSERVE PHASE FAILED - aborting autonomous session.")
            stats.stop_reason = StopReason.OBSERVE_FAILED.value
            stats.mode = RuntimeMode.OBSERVE.value
            stats.ticks = OBSERVE_TICKS
            return stats

        print(f"\n  OBSERVE phase PASSED. Enabling autonomous actions...")

        # ===================================================================
        # 4. AUTONOMOUS_ISOLATED phase
        # ===================================================================
        print(f"\n{SEP}")
        print(f"  PHASE 2: AUTONOMOUS_ISOLATED (max {MAX_RUNTIME_SECONDS}s / {MAX_TICKS} ticks)")
        print(SEP)

        # Enable physical actions and learning
        runtime.action_executor = action_executor
        runtime.learning_integrator = learning_integrator
        runtime.reset_state()

        stats.mode = RuntimeMode.AUTONOMOUS_ISOLATED.value
        stop_reason = StopReason.TIME_LIMIT

        tick_times: List[float] = []
        consecutive_failures = 0
        consecutive_same_action = 0
        last_action_type = None
        event_timestamps: List[float] = []
        total_rewards_by_domain: Dict[str, float] = {"combat": 0.0, "farm": 0.0, "roam": 0.0}
        event_counts: Dict[str, int] = Counter()

        session_start = time.perf_counter()
        last_checkpoint_wallclock = time.time()
        tick_num = 0

        try:
            while True:
                tick_num += 1
                t0 = time.perf_counter()

                # --- Check limits ---
                elapsed = time.perf_counter() - session_start
                if elapsed >= MAX_RUNTIME_SECONDS:
                    stop_reason = StopReason.TIME_LIMIT
                    break
                if tick_num >= MAX_TICKS:
                    stop_reason = StopReason.TICK_LIMIT
                    break

                # --- Execute tick ---
                result = runtime.tick()
                tick_ms = (time.perf_counter() - t0) * 1000.0
                tick_times.append(tick_ms)

                if not result.success:
                    consecutive_failures += 1
                    stats.actions_failed += 1
                    logger.warning(f"Tick {tick_num} FAILED: {result.error} "
                                   f"(consecutive={consecutive_failures})")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        stop_reason = StopReason.FAILURE_LIMIT
                        break
                    continue
                else:
                    consecutive_failures = 0

                # --- Track action ---
                action = result.action
                action_type = action.type.value if action else "idle"
                stats.actions_total += 1

                if result.execution_result and result.execution_result.success:
                    stats.actions_successful += 1
                else:
                    stats.actions_failed += 1

                stats.action_distribution[action_type] = stats.action_distribution.get(action_type, 0) + 1

                # Same-action watchdog
                if action_type == last_action_type:
                    consecutive_same_action += 1
                else:
                    consecutive_same_action = 1
                    last_action_type = action_type

                if consecutive_same_action == SAME_ACTION_WARN:
                    msg = f"WATCHDOG: {consecutive_same_action} consecutive '{action_type}' actions"
                    stats.watchdog_warnings.append(msg)
                    logger.warning(msg)

                if consecutive_same_action >= SAME_ACTION_STOP:
                    msg = f"WATCHDOG STOP: {consecutive_same_action} consecutive '{action_type}' actions"
                    stats.watchdog_warnings.append(msg)
                    stats.watchdog_stops += 1
                    logger.error(msg)
                    stop_reason = StopReason.WATCHDOG_SAME_ACTION
                    break

                # --- Track events ---
                if result.events:
                    # Keep every event for learning/statistics, but the safety
                    # watchdog ignores tracker-lifecycle notifications. A vision
                    # frame can legitimately produce ENTERED/LOST changes without
                    # representing a gameplay failure or action storm.
                    critical_event_seen = False
                    critical_event_types = {
                        EventType.HERO_KILL,
                        EventType.CREEP_KILL,
                        EventType.PLAYER_DEATH,
                    }
                    for event in result.events:
                        event_name = event.type.value if hasattr(event.type, "value") else str(event.type)
                        event_counts[event_name] += 1
                        stats.events_total += 1
                        if event.type in critical_event_types:
                            critical_event_seen = True

                    if critical_event_seen:
                        event_timestamps.append(time.time())

                # Event storm watchdog: too many critical EVENTFUL TICKS, not raw
                # tracker lifecycle events.
                now_time = time.time()
                recent_events = [t for t in event_timestamps if now_time - t < EVENT_STORM_WINDOW]
                if len(recent_events) > EVENT_STORM_THRESHOLD:
                    msg = (
                        f"WATCHDOG STOP: Event storm - {len(recent_events)} eventful ticks "
                        f"in {EVENT_STORM_WINDOW}s"
                    )
                    stats.watchdog_warnings.append(msg)
                    stats.watchdog_stops += 1
                    logger.error(msg)
                    stop_reason = StopReason.WATCHDOG_EVENT_STORM
                    break

                # --- Track rewards/transitions ---
                for transition in result.transitions:
                    stats.transitions_total += 1
                    stats.q_updates_total += 1

                # Build per-decision log record
                tactical = result.tactical_state
                state_desc = "?"
                if tactical:
                    if tactical.enemy_visible:
                        dist = f" d={tactical.nearest_enemy_distance:.0f}" if tactical.nearest_enemy_distance else ""
                        state_desc = f"COMBAT{dist} hp={tactical.player_hp_ratio:.2f}"
                    elif tactical.minion_count > 0:
                        state_desc = f"FARM m={tactical.minion_count} hp={tactical.player_hp_ratio:.2f}"
                    else:
                        state_desc = f"ROAM hp={tactical.player_hp_ratio:.2f}"

                direction_str = None
                if action and action.direction:
                    direction_str = f"({action.direction.x:.0f},{action.direction.y:.0f})"

                reward_list = []
                for tr in result.transitions:
                    reward_list.append({
                        "state": tr.state_key,
                        "action": tr.action,
                        "reward": tr.reward,
                        "done": tr.done,
                    })
                    # Track domain rewards
                    domain = isolated_ql.infer_domain(tr.state_key)
                    if tr.reward > 0:
                        stats.rewards_positive_total += tr.reward
                    else:
                        stats.rewards_negative_total += tr.reward
                    total_rewards_by_domain[domain] = total_rewards_by_domain.get(domain, 0.0) + tr.reward
                    stats.rewards_count += 1

                rec = DecisionRecord(
                    tick=tick_num,
                    timestamp=result.timestamp,
                    mode=RuntimeMode.AUTONOMOUS_ISOLATED.value,
                    state_summary=state_desc,
                    action_type=action_type,
                    action_direction=direction_str,
                    action_target=action.target_id if action else None,
                    execution_ok=bool(result.execution_result and result.execution_result.success),
                    execution_error=(result.execution_result.error
                                     if result.execution_result and not result.execution_result.success
                                     else None),
                    events=[e.type.value if hasattr(e.type, "value") else str(e.type) for e in result.events],
                    rewards=reward_list,
                    transitions=len(result.transitions),
                    q_updates=len(result.transitions),
                )
                session_log.append(rec)

                # Compact per-tick line
                events_str = ",".join(rec.events) if rec.events else "-"
                rew_str = f"r={len(reward_list)}" if reward_list else ""
                exec_str = "OK" if rec.execution_ok else f"FAIL:{rec.execution_error}"
                print(f"  #{tick_num:03d} | {tick_ms:5.1f}ms | {state_desc:30s} | "
                      f"{action_type:10s} | {exec_str:4s} | ev={events_str} {rew_str}")

                # Durable checkpoint for long isolated training.  This does NOT
                # touch production Q and makes Ctrl+C/watchdog stops recoverable.
                if time.time() - last_checkpoint_wallclock >= CHECKPOINT_INTERVAL_SECONDS:
                    try:
                        save_isolated_q(
                            isolated_ql,
                            ISOLATED_Q_PATH,
                            {
                                "stage": "29.1",
                                "session_duration": round(time.perf_counter() - session_start, 2),
                                "ticks": tick_num,
                                "transitions": stats.transitions_total,
                                "stop_reason": "CHECKPOINT",
                            },
                        )
                        save_session_log(session_log, SESSION_LOG_PATH)
                        last_checkpoint_wallclock = time.time()
                        print(f"  [checkpoint] isolated Q + session log saved at tick {tick_num}")
                    except Exception as e:
                        logger.warning(f"Checkpoint failed: {e}")

        except KeyboardInterrupt:
            stop_reason = StopReason.USER_INTERRUPT
            logger.info("User interrupt (Ctrl+C)")

        except Exception as e:
            stop_reason = StopReason.RUNTIME_ERROR
            logger.exception(f"Runtime error: {e}")

        finally:
            # ===================================================================
            # 5. Teardown (ALWAYS executes)
            # ===================================================================
            print(f"\n[5] Teardown...")
            elapsed_total = time.perf_counter() - session_start
            stats.duration_sec = round(elapsed_total, 2)
            stats.ticks = tick_num
            stats.stop_reason = stop_reason.value

            # Release joystick
            if joystick:
                try:
                    joystick.release()
                    stats.joystick_released = True
                    print("  Joystick released.")
                except Exception as e:
                    logger.warning(f"Joystick release failed: {e}")

            # Stop runtime (closes capture)
            try:
                runtime.stop()
                print("  Runtime stopped.")
            except Exception as e:
                logger.warning(f"Runtime stop error: {e}")

            # Close ADB
            if adb:
                try:
                    adb.close()
                    print("  ADB closed.")
                except Exception as e:
                    logger.warning(f"ADB close error: {e}")

            # Terminate Scrcpy
            if scrcpy_proc and scrcpy_proc.poll() is None:
                try:
                    scrcpy_proc.terminate()
                    scrcpy_proc.wait(timeout=3)
                    print("  Scrcpy terminated.")
                except Exception:
                    scrcpy_proc.kill()
                    print("  Scrcpy killed.")

            stats.teardown_ok = True
            print("  Teardown complete.")

            # Timing stats
            if tick_times:
                stats.avg_tick_ms = round(statistics.mean(tick_times), 2)
                stats.p95_tick_ms = round(pct(tick_times, 95), 2)

            # One-action dominance check
            if stats.actions_total > 0:
                max_action = max(stats.action_distribution.items(), key=lambda x: x[1], default=("idle", 0))
                dominance = max_action[1] / stats.actions_total
                if dominance > ONE_ACTION_DOMINANCE:
                    stats.dominant_action = f"{max_action[0]} ({dominance*100:.0f}%)"
                    msg = f"WARNING: One-action dominance: {stats.dominant_action}"
                    stats.watchdog_warnings.append(msg)

                # Repetition rate
                transitions_count = 0
                for i in range(1, len(session_log)):
                    if (session_log[i].mode == RuntimeMode.AUTONOMOUS_ISOLATED.value and
                        session_log[i].action_type == session_log[i-1].action_type):
                        transitions_count += 1
                auto_recs = [r for r in session_log if r.mode == RuntimeMode.AUTONOMOUS_ISOLATED.value]
                stats.repetition_rate = round(transitions_count / max(len(auto_recs) - 1, 1), 3)

            # Event counts
            stats.events_hero_kill = event_counts.get("hero_kill", 0) + event_counts.get("HERO_KILL", 0)
            stats.events_creep_kill = event_counts.get("creep_kill", 0) + event_counts.get("CREEP_KILL", 0)
            stats.events_player_death = event_counts.get("player_death", 0) + event_counts.get("PLAYER_DEATH", 0)
            stats.events_target_entered = event_counts.get("target_entered", 0) + event_counts.get("TARGET_ENTERED", 0)
            stats.events_target_lost = event_counts.get("target_lost", 0) + event_counts.get("TARGET_LOST", 0)

            # Reward domain totals
            stats.rewards_combat_total = round(total_rewards_by_domain.get("combat", 0.0), 2)
            stats.rewards_farm_total = round(total_rewards_by_domain.get("farm", 0.0), 2)
            stats.rewards_roam_total = round(total_rewards_by_domain.get("roam", 0.0), 2)

            # Q states after
            stats.q_states_after = count_q_states(isolated_ql)

            # Final isolated-Q save
            try:
                stats.isolated_q_sha = save_isolated_q(
                    isolated_ql,
                    ISOLATED_Q_PATH,
                    {
                        "stage": "29.1",
                        "session_duration": stats.duration_sec,
                        "ticks": stats.ticks,
                        "transitions": stats.transitions_total,
                        "stop_reason": stats.stop_reason,
                    },
                )
                stats.isolated_q_path = ISOLATED_Q_PATH
                print(f"  Isolated Q saved: {ISOLATED_Q_PATH}")
            except Exception as e:
                logger.warning(f"Failed to save isolated Q: {e}")

            # Verify production Q unchanged
            stats.q_brain_sha_after = sha256(Q_BRAIN_PATH)
            stats.baseline_sha_after = sha256(BASELINE_PATH)

            # Save session log as JSONL
            log_path = SESSION_LOG_PATH
            try:
                save_session_log(session_log, log_path)
                print(f"  Session log saved: {log_path}")
            except Exception as e:
                logger.warning(f"Failed to save session log: {e}")

    except Exception as e:
        logger.exception(f"Fatal setup error: {e}")
        stats.stop_reason = StopReason.RUNTIME_ERROR.value

    # ===================================================================
    # 6. Report
    # ===================================================================
    print(f"\n{SEP}")
    print("  STAGE 29.1 -- SESSION REPORT")
    print(SEP)

    q_ok = (stats.q_brain_sha_before == stats.q_brain_sha_after)
    b_ok = (stats.baseline_sha_before == stats.baseline_sha_after)

    print(f"""
  Runtime:
    Mode:              {stats.mode}
    Duration:          {stats.duration_sec:.1f} seconds
    Ticks:             {stats.ticks}
    Avg tick:          {stats.avg_tick_ms:.1f} ms
    P95 tick:          {stats.p95_tick_ms:.1f} ms
    Stop reason:       {stats.stop_reason}

  Actions:
    Total:             {stats.actions_total}
    Successful:        {stats.actions_successful}
    Failed:            {stats.actions_failed}
    Distribution:      {json.dumps(stats.action_distribution, indent=6)}
    Repetition rate:   {stats.repetition_rate:.3f}
    Dominant action:   {stats.dominant_action or 'None'}

  Events:
    Total:             {stats.events_total}
    HERO_KILL:         {stats.events_hero_kill}
    CREEP_KILL:        {stats.events_creep_kill}
    PLAYER_DEATH:      {stats.events_player_death}
    TARGET_ENTERED:    {stats.events_target_entered}
    TARGET_LOST:       {stats.events_target_lost}

  Rewards:
    Count:             {stats.rewards_count}
    Combat total:      {stats.rewards_combat_total:+.1f}
    Farm total:        {stats.rewards_farm_total:+.1f}
    Roam total:        {stats.rewards_roam_total:+.1f}
    Positive total:    {stats.rewards_positive_total:+.1f}
    Negative total:    {stats.rewards_negative_total:+.1f}

  Learning:
    Transitions:       {stats.transitions_total}
    Q updates:         {stats.q_updates_total}
    Q states before:   {stats.q_states_before}
    Q states after:    {stats.q_states_after}
    Isolated Q file:   {stats.isolated_q_path}
    Isolated Q SHA:    {stats.isolated_q_sha[:24]}...

  Safety:
    Watchdog warnings: {len(stats.watchdog_warnings)}
    Watchdog stops:    {stats.watchdog_stops}
    Emergency stop:    {stats.emergency_stop}
    Joystick released: {stats.joystick_released}
    Teardown OK:       {stats.teardown_ok}

  Q-Memory Integrity:
    q_brain.json:      {"UNCHANGED [OK]" if q_ok else "CHANGED [FAIL]"} {stats.q_brain_sha_after[:16]}
    q_brain_baseline:  {"UNCHANGED [OK]" if b_ok else "CHANGED [FAIL]"} {stats.baseline_sha_after[:16]}
""")

    if stats.watchdog_warnings:
        print("  Watchdog details:")
        for w in stats.watchdog_warnings:
            print(f"    - {w}")

    if not q_ok:
        print("\n  *** CRITICAL: Production q_brain.json was MODIFIED! ***")
    if not b_ok:
        print("\n  *** CRITICAL: q_brain_baseline.json was MODIFIED! ***")

    # Save stats summary
    stats_path = SUMMARY_PATH
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(asdict(stats), f, indent=2, ensure_ascii=False)
        print(f"\n  Summary saved: {stats_path}")
    except Exception as e:
        logger.warning(f"Failed to save summary: {e}")

    print(f"\n{SEP}")
    print("  STAGE 29.1 SESSION COMPLETE")
    print(SEP)

    return stats


if __name__ == "__main__":
    main()
