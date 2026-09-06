"""
tools/real_device_smoke.py — Stage 23B Real Device Smoke Validation.

Validates the real physical V2 pipeline against a live Android device + bundled Scrcpy:
- Scrcpy (tools/scrcpy) -> ScreenCapture -> real BGR frame (1544x720)
- Vision: YOLOv8 (models/yolo/v4/best.pt) + SkillStateChecker + HPDetector
- World: WorldTracker + WorldStateBuilder -> WorldState
- Events: WorldEventDetector + WatchdogManager
- Control: JoystickControl + AttackControl + SkillsControl
- ActionExecutor: Action -> ActionExecutor -> Control -> ADBTransport -> Android
- Brain: TacticalState -> ClaudeBrainV2 -> Action (isolated learning)
- Runtime: Bounded V2Runtime execution (max_ticks = 5)
- Emergency Stop: Joystick release + touch reset + clean shutdown
- Persistence Safety: Zero mutation of production q_brain.json / baseline
"""

import os
import sys
import time
import subprocess
import hashlib
from typing import Optional, List, Dict, Any

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from config.config import (
    PROJECT_ROOT,
    SCRCPY_PATH,
    SCRCPY_DIR,
    ADB_PATH,
    DEVICE_SERIAL,
    WEIGHTS_PATH,
)
from actions.executor import ActionExecutor
from actions.models import Action, ActionType
from app.runtime import V2Runtime, RuntimeTickResult
from brain.brain import ClaudeBrainV2
from brain.q_learning import QLearningCore
from brain.rewards import RewardCalculator
from brain.memory import BrainMemory
from brain.state import TacticalStateBuilder, TacticalState
from control.adb import ADBTransport
from control.attack import AttackControl
from control.humanizer import InputHumanizer
from control.joystick import JoystickControl
from control.skills import SkillsControl
from vision.capture import ScreenCapture
from vision.detector import YoloDetector, Detection
from vision.hp_detector import HPDetector, HPObservation
from vision.skill_state import SkillStateChecker, SkillState
from world.builder import WorldStateBuilder
from world.events import WorldEventDetector, EventType, GameEvent
from world.models import Vector2, WorldState
from world.tracker import WorldTracker


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class HPDetectorAdapter:
    """Adapts HPDetector to provide a .detect(frame) method for V2Runtime."""
    def __init__(self, hp_detector: HPDetector):
        self._detector = hp_detector

    def detect(self, frame) -> HPObservation:
        # Default estimation at canonical hero screen center (772, 360)
        return self._detector.detect_player_hp(frame, (772.0, 360.0))


def run_stage_23b_smoke():
    print("=" * 70)
    print("  STAGE 23B — REAL DEVICE FUNCTIONAL SMOKE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 0. Persistence Baseline Record
    # --------------------------------------------------------------------------
    baseline_path = os.path.join(PROJECT_ROOT, "data", "q_brain_baseline.json")
    q_brain_path = os.path.join(PROJECT_ROOT, "q_brain.json")
    init_baseline_hash = file_sha256(baseline_path)
    init_q_hash = file_sha256(q_brain_path)
    print(f"[*] Pre-flight q_brain_baseline.json SHA-256: {init_baseline_hash[:16]}...")
    print(f"[*] Pre-flight q_brain.json SHA-256:          {init_q_hash[:16]}...")

    # --------------------------------------------------------------------------
    # 1. Launch Bundled Scrcpy Process
    # --------------------------------------------------------------------------
    print("\n--- Phase 0: Launching Bundled Scrcpy ---")
    subprocess.run(["taskkill", "/F", "/IM", "scrcpy.exe"], capture_output=True)
    time.sleep(0.5)

    scrcpy_cmd = [SCRCPY_PATH, "-s", DEVICE_SERIAL, "--window-title", "scrcpy"]
    print(f"[*] Command: {' '.join(scrcpy_cmd)}")
    scrcpy_proc = subprocess.Popen(
        scrcpy_cmd,
        cwd=SCRCPY_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[*] Scrcpy launched with PID: {scrcpy_proc.pid}")

    # Wait for Scrcpy texture/display confirmation
    t_wait = time.time()
    texture_ready = False
    while time.time() - t_wait < 5.0:
        line = scrcpy_proc.stdout.readline()
        if line:
            l_str = line.strip()
            if "device" in l_str.lower() or "direct3d" in l_str.lower() or "texture" in l_str.lower():
                print(f"  [scrcpy] {l_str}")
            if "Texture:" in l_str:
                texture_ready = True
                break
        if scrcpy_proc.poll() is not None:
            break

    # --------------------------------------------------------------------------
    # 2. ScreenCapture: Window Discovery & Frame Grab
    # --------------------------------------------------------------------------
    print("\n--- Phase 1: Real ScreenCapture Verification ---")
    capture = ScreenCapture()
    hwnd = capture.find_scrcpy_window()
    print(f"[*] ScreenCapture.find_scrcpy_window() -> HWND: {hwnd}")
    if not hwnd:
        scrcpy_proc.terminate()
        raise RuntimeError("FAIL: Could not locate Scrcpy window handle!")

    # Grab first real frame
    t0 = time.perf_counter()
    real_frame = capture.grab(hwnd=hwnd)
    grab_ms = (time.perf_counter() - t0) * 1000.0

    if real_frame is None:
        scrcpy_proc.terminate()
        raise RuntimeError("FAIL: ScreenCapture.grab() returned None!")

    print(f"[*] ScreenCapture.grab() SUCCESS -> shape: {real_frame.shape}, latency: {grab_ms:.2f}ms")
    assert real_frame.shape == (720, 1544, 3), f"Unexpected frame shape: {real_frame.shape}"
    print(f"[*] Frame stats: min={real_frame.min()}, max={real_frame.max()}, mean={real_frame.mean():.2f}")

    # --------------------------------------------------------------------------
    # 3. Real Vision: YOLO Inference & Sensors
    # --------------------------------------------------------------------------
    print("\n--- Phase 2: Real Vision Inference on Live Frame ---")
    detector = YoloDetector(weights_path=WEIGHTS_PATH)
    print(f"[*] YOLO loaded from: {WEIGHTS_PATH}")
    print(f"[*] Model class names: {detector.names}")

    t0 = time.perf_counter()
    detections = detector.detect(real_frame)
    yolo_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[*] YOLO inference completed in {yolo_ms:.2f}ms. Total detections: {len(detections)}")

    for i, d in enumerate(detections):
        print(f"  Det #{i+1}: class='{d.class_name}' ({d.class_id}), conf={d.confidence:.2f}, "
              f"center=({d.center[0]:.1f}, {d.center[1]:.1f}), bbox=[{d.x1:.1f}, {d.y1:.1f}, {d.x2:.1f}, {d.y2:.1f}]")
        # Validate coordinates & values
        assert not (d.x1 != d.x1 or d.y1 != d.y1 or d.x2 != d.x2 or d.y2 != d.y2), "NaN detected in bbox!"
        assert 0.0 <= d.confidence <= 1.0, f"Invalid confidence: {d.confidence}"
        assert 0.0 <= d.x1 <= 1544.0 and 0.0 <= d.x2 <= 1544.0, f"X coords out of range: {d.x1}, {d.x2}"
        assert 0.0 <= d.y1 <= 720.0 and 0.0 <= d.y2 <= 720.0, f"Y coords out of range: {d.y1}, {d.y2}"

    # Skill and HP sensors
    skill_checker = SkillStateChecker()
    skill_state = skill_checker.check(real_frame)
    print(f"[*] Skill readiness: S1={skill_state.s1_ready}, S2={skill_state.s2_ready}, Ult={skill_state.ultimate_ready}")

    hp_detector = HPDetector()
    hp_obs = hp_detector.detect_player_hp(real_frame, (772.0, 360.0))
    print(f"[*] Player HP sensor: value={hp_obs.value:.2f}, conf={hp_obs.confidence:.2f}, visible={hp_obs.visible}")

    # --------------------------------------------------------------------------
    # 4. Live WorldState & Tracker Feeding
    # --------------------------------------------------------------------------
    print("\n--- Phase 3: Feeding Vision into Live WorldState & Tracker ---")
    now = time.time()
    tracker = WorldTracker()
    tracks = tracker.update(detections, now)
    print(f"[*] WorldTracker active tracks: {len(tracks)}")

    world_builder = WorldStateBuilder()
    world_state = world_builder.build(
        timestamp=now,
        detections=detections,
        tracks=tracks,
        player_hp=hp_obs,
        skills=skill_state,
    )
    print(f"[*] Live WorldState constructed:")
    print(f"    Player visible: {world_state.player.is_visible if world_state.player else False}")
    print(f"    Player dead:    {world_state.player.is_dead if world_state.player else False}")
    print(f"    Enemies count:  {len(world_state.enemies)}")
    print(f"    Minions count:  {len(world_state.minions)}")
    print(f"    Turrets count:  {len(world_state.turrets)}")

    # Semantic check: LOST != DEAD
    if world_state.player and not world_state.player.is_visible:
        assert not world_state.player.is_dead, "CRITICAL: Player marked dead purely from missing visibility!"
    print("[*] Semantic assertion passed: LOST != DEAD strictly preserved.")

    # --------------------------------------------------------------------------
    # 5. Live World Events & Watchdog
    # --------------------------------------------------------------------------
    print("\n--- Phase 4: Live Event Detector & Watchdogs ---")
    event_detector = WorldEventDetector()
    events = event_detector.update(
        world_state=world_state,
        tracks=tracks,
        timestamp=now,
    )
    print(f"[*] Events fired on initial live frame: {[e.type.name for e in events]}")
    # Assert no fake kills from initial frame
    assert not any(e.type == EventType.HERO_KILL for e in events), "False HERO_KILL emitted on initial frame!"
    assert not any(e.type == EventType.PLAYER_DEATH for e in events), "False PLAYER_DEATH emitted on initial frame!"
    print("[*] Event assertion passed: No false kill/death generated.")

    # --------------------------------------------------------------------------
    # 6. Controlled Physical Actions via ActionExecutor & ADBTransport
    # --------------------------------------------------------------------------
    print("\n--- Phase 5: Controlled Physical Actions via ActionExecutor ---")
    adb_transport = ADBTransport(binary=ADB_PATH, device_serial=DEVICE_SERIAL, auto_start=True)
    print(f"[*] ADBTransport connected: {adb_transport.is_alive()}")

    humanizer = InputHumanizer()
    joystick = JoystickControl(transport=adb_transport, time_provider=time.time)
    attack_ctrl = AttackControl(transport=adb_transport, humanizer=humanizer, joystick=joystick, time_provider=time.time)
    skills_ctrl = SkillsControl(transport=adb_transport, humanizer=humanizer, joystick=joystick)

    action_executor = ActionExecutor(
        joystick=joystick,
        skills=skills_ctrl,
        attack_controller=attack_ctrl,
        hero_pos_provider=lambda: (772.0, 360.0),
    )

    # 6A. Short controlled MOVE
    print("[*] Dispatching controlled low-risk MOVE...")
    move_action = Action(type=ActionType.MOVE, direction=Vector2(772.0 + 30.0, 360.0 + 20.0), duration=0.1)
    res_move = action_executor.execute(move_action)
    print(f"    MOVE execution result: success={res_move.success}, action={res_move.action.type.name}")
    assert res_move.success, f"MOVE failed: {res_move.error}"

    # 6B. Release joystick
    print("[*] Dispatching explicit joystick release...")
    rel_ok = joystick.release()
    print(f"    Joystick release success: {rel_ok}")

    # 6C. Controlled basic ATTACK
    print("[*] Dispatching controlled basic ATTACK...")
    time.sleep(0.25)  # Respect attack cooldown
    attack_action = Action(type=ActionType.ATTACK)
    res_attack = action_executor.execute(attack_action)
    print(f"    ATTACK execution result: success={res_attack.success}, action={res_attack.action.type.name}")
    assert res_attack.success, f"ATTACK failed: {res_attack.error}"

    # 6D. Controlled CAST_S1
    print("[*] Dispatching controlled CAST_S1...")
    time.sleep(0.25)
    s1_action = Action(type=ActionType.CAST_S1)
    res_s1 = action_executor.execute(s1_action)
    print(f"    CAST_S1 execution result: success={res_s1.success}, action={res_s1.action.type.name}")
    assert res_s1.success, f"CAST_S1 failed: {res_s1.error}"

    # --------------------------------------------------------------------------
    # 7. Live Brain Decision (Learning Isolated)
    # --------------------------------------------------------------------------
    print("\n--- Phase 6: Live Brain Decision (ClaudeBrainV2) ---")
    tactical_builder = TacticalStateBuilder()
    tactical_state = tactical_builder.build(world_state)
    print(f"[*] TacticalState: enemy_visible={tactical_state.enemy_visible}, "
          f"minion_count={tactical_state.minion_count}, "
          f"can_combo={tactical_state.can_combo}")

    # Isolated Q-learning and memory instance for testing decisions without disk writes
    test_ql = QLearningCore()
    test_memory = BrainMemory()
    brain = ClaudeBrainV2(q_learning=test_ql, memory=test_memory)
    live_action = brain.decide(tactical_state)
    print(f"[*] ClaudeBrainV2 selected Action: type={live_action.type.name}, direction={live_action.direction}")
    assert isinstance(live_action, Action), f"Expected Action object, got {type(live_action)}"

    # --------------------------------------------------------------------------
    # 8. Complete V2Runtime Bounded Execution (max_ticks = 5)
    # --------------------------------------------------------------------------
    print("\n--- Phase 7: Bounded V2Runtime Live Loop (max_ticks = 5) ---")
    runtime = V2Runtime(
        capture=capture,
        detector=detector,
        world_builder=world_builder,
        tracker=tracker,
        event_detector=event_detector,
        tactical_state_builder=tactical_builder,
        brain=brain,
        action_executor=action_executor,
        learning_integrator=None,  # Keep production learning strictly disabled
        hp_detector=HPDetectorAdapter(hp_detector),
        skill_checker=skill_checker,
        time_provider=time.time,
    )

    tick_count = 0
    t_runtime_start = time.perf_counter()
    while tick_count < 5:
        t_tick_0 = time.perf_counter()
        tick_res: RuntimeTickResult = runtime.tick()
        t_tick_ms = (time.perf_counter() - t_tick_0) * 1000.0
        tick_count += 1
        print(f"  Tick #{tick_count}: success={tick_res.success}, frame_captured={tick_res.frame_captured}, "
              f"action={tick_res.action.type.name if tick_res.action else None}, "
              f"events={len(tick_res.events)}, tick_latency={t_tick_ms:.1f}ms")
        assert tick_res.success, f"Runtime tick #{tick_count} failed: {tick_res.error}"
        assert tick_res.frame_captured, f"Frame not captured on tick #{tick_count}"
        time.sleep(0.05)

    tot_runtime_ms = (time.perf_counter() - t_runtime_start) * 1000.0
    print(f"[*] Completed {tick_count} live runtime ticks in {tot_runtime_ms:.1f}ms (avg {tot_runtime_ms/tick_count:.1f}ms/tick)")

    # --------------------------------------------------------------------------
    # 9. Minimal Isolated Learning Sample (Zero Production Mutation)
    # --------------------------------------------------------------------------
    print("\n--- Phase 8: Isolated Learning Sample Verification ---")
    from brain.encoder import TacticalStateEncoder
    from brain.transition_builder import TransitionBuilder
    from brain.learning_loop import LearningLoop
    from brain.learning_integration import LearningIntegrator

    isolated_encoder = TacticalStateEncoder()
    isolated_trans_builder = TransitionBuilder(isolated_encoder)
    isolated_ql = QLearningCore()
    isolated_loop = LearningLoop(
        encoder=isolated_encoder,
        transition_builder=isolated_trans_builder,
        q_learning=isolated_ql,
    )
    isolated_integrator = LearningIntegrator(
        reward_calculator=RewardCalculator(),
        transition_builder=isolated_trans_builder,
        learning_loop=isolated_loop,
    )

    # Process simulated event through isolated integrator
    sample_event = GameEvent(EventType.HERO_KILL, timestamp=now, entity_id=1, confidence=1.0, evidence=("kill_confirmed",))
    sample_transitions = isolated_integrator.process(
        state=tactical_state,
        action=move_action,
        events=[sample_event],
        next_state=tactical_state,
        done=False,
    )
    print(f"[*] Isolated learning executed: {len(sample_transitions)} transitions generated.")
    for tr in sample_transitions:
        print(f"    Transition: state={tr.state_key}, action={tr.action}, reward={tr.reward}")
    assert len(sample_transitions) == 2, "Expected 2 domain transitions for HERO_KILL (combat + roam)!"

    # --------------------------------------------------------------------------
    # 10. Emergency Stop & Cleanup
    # --------------------------------------------------------------------------
    print("\n--- Phase 9: Emergency Stop & Teardown ---")
    runtime.stop()
    print("[*] V2Runtime stopped.")
    joystick.release()
    print("[*] Joystick touch explicitly released.")
    adb_transport.stop()
    print("[*] ADBTransport session closed.")
    capture.close()
    print("[*] ScreenCapture closed.")
    scrcpy_proc.terminate()
    try:
        scrcpy_proc.wait(timeout=2.0)
    except Exception:
        scrcpy_proc.kill()
    print("[*] Scrcpy process terminated cleanly.")

    # --------------------------------------------------------------------------
    # 11. Persistence Safety & Hash Verification
    # --------------------------------------------------------------------------
    print("\n--- Phase 10: Persistence Safety Verification ---")
    final_baseline_hash = file_sha256(baseline_path)
    final_q_hash = file_sha256(q_brain_path)
    assert init_baseline_hash == final_baseline_hash, "CRITICAL: data/q_brain_baseline.json was modified!"
    assert init_q_hash == final_q_hash, "CRITICAL: q_brain.json was modified during smoke test!"
    print(f"[*] Verified q_brain_baseline.json unchanged: {final_baseline_hash[:16]}... (MATCH)")
    print(f"[*] Verified q_brain.json unchanged:          {final_q_hash[:16]}... (MATCH)")

    print("\n" + "=" * 70)
    print("  STAGE 23B PHYSICAL SMOKE VALIDATION: ALL CHECKS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    run_stage_23b_smoke()
