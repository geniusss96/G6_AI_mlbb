"""
tools/inference_policy_bench.py -- Stage 27: Production Inference Policy Benchmark.

Runs the IDENTICAL V2 runtime pipeline twice on the same physical device:
  Pass 1: YOLO_DEVICE = "cpu"    (explicit CPU)
  Pass 2: YOLO_DEVICE = "cuda:0" (explicit GPU)

Measures per-component latency for both. Decides and documents the production default.

Constraints:
  - No q_brain.json writes (isolated QLearningCore).
  - No autonomous combat.
  - SHA-256 of q_brain.json + baseline verified before/after.
  - WARM_TICKS >= 30 for each mode.
  - Monotonic timing (time.perf_counter) throughout.
  - Between-mode teardown: scrcpy NOT restarted (capture path identical).
    Detector is rebuilt for each mode; all other components are shared.
"""

import os
import sys
import time
import subprocess
import hashlib
import statistics
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

WARM_TICKS = 30
Q_BRAIN_PATH   = os.path.join(WORKSPACE_ROOT, "q_brain.json")
BASELINE_PATH  = os.path.join(WORKSPACE_ROOT, "data", "q_brain_baseline.json")

SEP  = "=" * 72
SEP2 = "-" * 60


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


def stats(data: List[float]) -> dict:
    if not data:
        return {}
    return {
        "avg": round(statistics.mean(data), 2),
        "min": round(min(data), 2),
        "max": round(max(data), 2),
        "p50": round(pct(data, 50), 2),
        "p95": round(pct(data, 95), 2),
        "n":   len(data),
    }


def fmt(label: str, d: dict, unit: str = "ms") -> str:
    return (
        f"  {label:<22}: avg={d['avg']:.1f}{unit}  "
        f"min={d['min']:.1f}  max={d['max']:.1f}  "
        f"p50={d['p50']:.1f}  p95={d['p95']:.1f}  n={d['n']}"
    )


# ---------------------------------------------------------------------------
# Per-tick record
# ---------------------------------------------------------------------------
@dataclass
class Tick:
    idx: int
    capture_ms:    float = 0.0
    vision_ms:     float = 0.0
    track_ms:      float = 0.0
    hp_ms:         float = 0.0
    skill_ms:      float = 0.0
    world_ms:      float = 0.0
    event_ms:      float = 0.0
    tactical_ms:   float = 0.0
    brain_ms:      float = 0.0
    action_ms:     float = 0.0
    total_ms:      float = 0.0
    detections:    int   = 0
    success:       bool  = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Instrumented pipeline tick (same structure as perf_benchmark.py)
# ---------------------------------------------------------------------------
def run_tick(
    capture, detector, tracker,
    hp_adapter, skill_checker,
    world_builder, event_detector, tactical_builder,
    brain, action_executor,
    hwnd, idx: int,
) -> Tick:
    t = Tick(idx=idx)
    t0_total = time.perf_counter()

    # 1. Capture
    t0 = time.perf_counter()
    frame = capture.grab(hwnd=hwnd)
    t.capture_ms = (time.perf_counter() - t0) * 1000.0
    if frame is None:
        t.success = False; t.error = "Capture None"
        t.total_ms = (time.perf_counter() - t0_total) * 1000.0
        return t

    # 2. YOLO inference
    t0 = time.perf_counter()
    dets = detector.detect(frame)
    t.vision_ms = (time.perf_counter() - t0) * 1000.0
    t.detections = len(dets)

    # 3. Tracker
    t0 = time.perf_counter()
    tracks = tracker.update(dets, time.time())
    t.track_ms = (time.perf_counter() - t0) * 1000.0

    # 4. HP sensor
    t0 = time.perf_counter()
    player_hp = hp_adapter.detect(frame)
    t.hp_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Skill sensor
    t0 = time.perf_counter()
    skills = skill_checker.check(frame)
    t.skill_ms = (time.perf_counter() - t0) * 1000.0

    # 6. WorldState
    t0 = time.perf_counter()
    world = world_builder.build(
        timestamp=time.time(), detections=dets,
        tracks=tracks, player_hp=player_hp, skills=skills,
    )
    t.world_ms = (time.perf_counter() - t0) * 1000.0

    # 7. Events
    t0 = time.perf_counter()
    events = event_detector.update(world_state=world, tracks=tracks, timestamp=time.time())
    t.event_ms = (time.perf_counter() - t0) * 1000.0

    # 8. TacticalState
    t0 = time.perf_counter()
    tactical = tactical_builder.build(world)
    t.tactical_ms = (time.perf_counter() - t0) * 1000.0

    # 9. Brain
    t0 = time.perf_counter()
    action = brain.decide(tactical)
    t.brain_ms = (time.perf_counter() - t0) * 1000.0
    if action is None:
        from actions.models import Action, ActionType
        action = Action(type=ActionType.IDLE)

    # 10. ActionExecutor
    t0 = time.perf_counter()
    action_executor.execute(action)
    t.action_ms = (time.perf_counter() - t0) * 1000.0

    t.total_ms = (time.perf_counter() - t0_total) * 1000.0
    return t


# ---------------------------------------------------------------------------
# Single-mode benchmark pass
# ---------------------------------------------------------------------------
def run_pass(
    label: str,
    device: str,
    capture, hwnd,
    shared_components: dict,
    adb_path: str,
    device_serial: str,
) -> dict:
    """
    Runs WARM_TICKS instrumented ticks with the given YOLO device.
    Rebuilds YoloDetector for the requested device.
    Returns a results dict.
    """
    print(f"\n{SEP}")
    print(f"  PASS: {label}  (device='{device}')")
    print(SEP)

    # Import detector fresh for this device
    # Override YOLO_DEVICE env before constructing detector
    from vision.detector import YoloDetector
    from config.config import WEIGHTS_PATH, YOLO_IMGSZ

    import torch
    cuda_available = torch.cuda.is_available()
    cuda_strict = device.startswith("cuda") and cuda_available
    # For CPU pass, explicitly pass cuda_strict=False
    if device == "cpu":
        cuda_strict_flag = False
    else:
        cuda_strict_flag = cuda_available  # strict only if CUDA actually present

    print(f"  Building YoloDetector(device={device!r}, cuda_strict={cuda_strict_flag})...")
    t0 = time.perf_counter()
    detector = YoloDetector(
        weights_path=WEIGHTS_PATH,
        device=device,
        imgsz=YOLO_IMGSZ,
        cuda_strict=cuda_strict_flag,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Model load: {load_ms:.0f} ms")

    # First (cold) inference — triggers JIT/kernel warmup
    import numpy as np
    frame_probe = capture.grab(hwnd=hwnd)
    assert frame_probe is not None

    print("  Cold inference (JIT warmup)...")
    t0 = time.perf_counter()
    _ = detector.detect(frame_probe)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Cold inference: {cold_ms:.1f} ms")

    # Verify actual parameter device after first predict
    try:
        param_dev = str(next(detector.model.model.parameters()).device)
    except Exception:
        param_dev = "unknown"
    print(f"  Actual parameter device after first predict: {param_dev}")

    # GPU-specific: measure CUDA synchronize overhead on 5 standalone calls
    sync_overhead_ms = None
    if device.startswith("cuda") and cuda_available:
        try:
            sync_times = []
            for _ in range(5):
                frame_s = capture.grab(hwnd=hwnd)
                if frame_s is None:
                    break
                t0 = time.perf_counter()
                _ = detector.detect(frame_s)
                t_before_sync = time.perf_counter()
                torch.cuda.synchronize()
                t_after_sync = time.perf_counter()
                sync_times.append((t_after_sync - t_before_sync) * 1000.0)
            if sync_times:
                sync_overhead_ms = statistics.mean(sync_times)
                print(f"  CUDA synchronize overhead (avg over 5): {sync_overhead_ms:.2f} ms")
        except Exception as e:
            print(f"  [!] Could not measure sync overhead: {e}")

    # 2nd + 3rd standalone warm inferences (isolated, before full tick loop)
    warm_standalone = []
    for i in range(3):
        f = capture.grab(hwnd=hwnd)
        if f is None:
            break
        t0 = time.perf_counter()
        _ = detector.detect(f)
        warm_standalone.append((time.perf_counter() - t0) * 1000.0)
        print(f"  Standalone warm inference #{i+1}: {warm_standalone[-1]:.2f} ms")

    # -------------------------------------------------------------------
    # Full WARM_TICKS pipeline ticks
    # -------------------------------------------------------------------
    print(f"\n  Running {WARM_TICKS} warm pipeline ticks...")
    print(f"  {'Tick':>4} | {'Total':>6} | {'Cap':>5} | {'YOLO':>6} | "
          f"{'Trk':>4} | {'HP':>4} | {'Sk':>4} | {'Wld':>4} | "
          f"{'Evt':>4} | {'Brn':>4} | {'ADB':>5} | Det")
    print("  " + "-" * 78)

    ticks: List[Tick] = []
    for i in range(WARM_TICKS):
        t = run_tick(
            capture=capture,
            detector=detector,
            tracker=shared_components["tracker"],
            hp_adapter=shared_components["hp_adapter"],
            skill_checker=shared_components["skill_checker"],
            world_builder=shared_components["world_builder"],
            event_detector=shared_components["event_detector"],
            tactical_builder=shared_components["tactical_builder"],
            brain=shared_components["brain"],
            action_executor=shared_components["action_executor"],
            hwnd=hwnd,
            idx=i + 1,
        )
        ticks.append(t)
        status = "OK" if t.success else "FAIL"
        print(
            f"  #{i+1:02d}   | {t.total_ms:6.1f} | {t.capture_ms:5.1f} | "
            f"{t.vision_ms:6.1f} | {t.track_ms:4.2f} | {t.hp_ms:4.1f} | "
            f"{t.skill_ms:4.1f} | {t.world_ms:4.2f} | {t.event_ms:4.2f} | "
            f"{t.brain_ms:4.2f} | {t.action_ms:5.1f} | {t.detections} [{status}]"
        )
        if not t.success:
            print(f"  [!] Tick #{i+1} failed: {t.error}")

    good = [t for t in ticks if t.success]
    if not good:
        print("  [FATAL] No successful ticks!")
        return {"label": label, "device": device, "error": "no successful ticks"}

    # Statistics
    total_s   = stats([t.total_ms    for t in good])
    capture_s = stats([t.capture_ms  for t in good])
    vision_s  = stats([t.vision_ms   for t in good])
    track_s   = stats([t.track_ms    for t in good])
    hp_s      = stats([t.hp_ms       for t in good])
    skill_s   = stats([t.skill_ms    for t in good])
    world_s   = stats([t.world_ms    for t in good])
    event_s   = stats([t.event_ms    for t in good])
    tactical_s= stats([t.tactical_ms for t in good])
    brain_s   = stats([t.brain_ms    for t in good])
    action_s  = stats([t.action_ms   for t in good])

    avg_fps = 1000.0 / total_s["avg"] if total_s["avg"] > 0 else 0.0

    print(f"\n  --- {label} STATISTICS ({len(good)}/{WARM_TICKS} ticks) ---")
    print(fmt("Total tick", total_s))
    print(fmt("  Capture", capture_s))
    print(fmt("  Vision/YOLO", vision_s))
    print(fmt("  Tracking", track_s))
    print(fmt("  HP sensor", hp_s))
    print(fmt("  Skill sensor", skill_s))
    print(fmt("  WorldState", world_s))
    print(fmt("  Events", event_s))
    print(fmt("  TacticalState", tactical_s))
    print(fmt("  Brain", brain_s))
    print(fmt("  ActionExecutor", action_s))
    print(f"  {'Effective FPS':<22}: {avg_fps:.1f} FPS")

    result = {
        "label":         label,
        "device":        device,
        "param_device":  param_dev,
        "load_ms":       round(load_ms, 1),
        "cold_ms":       round(cold_ms, 1),
        "sync_overhead_ms": round(sync_overhead_ms, 2) if sync_overhead_ms is not None else None,
        "standalone_warm_ms": [round(x, 2) for x in warm_standalone],
        "ticks_total":   WARM_TICKS,
        "ticks_ok":      len(good),
        "total":         total_s,
        "capture":       capture_s,
        "vision":        vision_s,
        "track":         track_s,
        "hp":            hp_s,
        "skill":         skill_s,
        "world":         world_s,
        "event":         event_s,
        "tactical":      tactical_s,
        "brain":         brain_s,
        "action":        action_s,
        "fps":           round(avg_fps, 1),
    }

    # Cleanup detector explicitly (free VRAM before next pass)
    del detector
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Force UTF-8 stdout
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(SEP)
    print("  STAGE 27 -- CPU vs GPU INFERENCE POLICY BENCHMARK")
    print(SEP)

    # Pre-flight Q-memory hashes
    pre_q = sha256(Q_BRAIN_PATH)
    pre_b = sha256(BASELINE_PATH)
    print(f"\n  PRE q_brain.json SHA-256:     {pre_q[:24]}...")
    print(f"  PRE q_brain_baseline SHA-256: {pre_b[:24]}...")

    from config.config import (
        PROJECT_ROOT, SCRCPY_PATH, SCRCPY_DIR, ADB_PATH,
        DEVICE_SERIAL, WEIGHTS_PATH,
    )

    # -----------------------------------------------------------------------
    # Launch Scrcpy ONCE — shared for both passes
    # -----------------------------------------------------------------------
    print("\n[1] Launching Scrcpy (shared for both passes)...")
    subprocess.run(["taskkill", "/F", "/IM", "scrcpy.exe"], capture_output=True)
    time.sleep(0.5)

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
    print(f"  Texture ready: {texture_ready}")

    from vision.capture import ScreenCapture
    time.sleep(0.5)
    capture = ScreenCapture()
    hwnd = capture.find_scrcpy_window()
    if not hwnd:
        scrcpy_proc.terminate()
        print("FATAL: Could not find Scrcpy window.")
        sys.exit(1)
    print(f"  HWND={hwnd}")

    # Verify first frame
    first_frame = capture.grab(hwnd=hwnd)
    assert first_frame is not None and first_frame.shape == (720, 1544, 3), \
        f"Bad frame: {first_frame.shape if first_frame is not None else None}"
    print(f"  First frame shape: {first_frame.shape}")

    # -----------------------------------------------------------------------
    # Build shared V2 components (Brain with isolated Q, no production writes)
    # -----------------------------------------------------------------------
    print("\n[2] Building shared V2 components (isolated brain)...")
    from world.tracker import WorldTracker
    from world.builder import WorldStateBuilder
    from world.events import WorldEventDetector
    from brain.state import TacticalStateBuilder
    from brain.q_learning import QLearningCore
    from brain.memory import BrainMemory
    from brain.brain import ClaudeBrainV2
    from vision.hp_detector import HPDetector
    from vision.skill_state import SkillStateChecker
    from control.adb import ADBTransport
    from control.humanizer import InputHumanizer
    from control.joystick import JoystickControl
    from control.attack import AttackControl
    from control.skills import SkillsControl
    from actions.executor import ActionExecutor

    test_ql     = QLearningCore()
    test_memory = BrainMemory()
    brain       = ClaudeBrainV2(q_learning=test_ql, memory=test_memory)
    tracker         = WorldTracker()
    world_builder   = WorldStateBuilder()
    event_detector  = WorldEventDetector()
    tactical_builder = TacticalStateBuilder()

    hp_raw = HPDetector()
    class _HP:
        def detect(self, f): return hp_raw.detect_player_hp(f, (772.0, 360.0))
    hp_adapter = _HP()
    skill_checker = SkillStateChecker()

    adb = ADBTransport(binary=ADB_PATH, device_serial=DEVICE_SERIAL, auto_start=True)
    print(f"  ADB connected: {adb.is_alive()}")

    humanizer   = InputHumanizer()
    joystick    = JoystickControl(transport=adb, time_provider=time.time)
    attack_ctrl = AttackControl(transport=adb, humanizer=humanizer,
                                joystick=joystick, time_provider=time.time)
    skills_ctrl = SkillsControl(transport=adb, humanizer=humanizer, joystick=joystick)
    action_executor = ActionExecutor(
        joystick=joystick,
        skills=skills_ctrl,
        attack_controller=attack_ctrl,
        hero_pos_provider=lambda: (772.0, 360.0),
    )

    shared = {
        "tracker": tracker, "hp_adapter": hp_adapter,
        "skill_checker": skill_checker, "world_builder": world_builder,
        "event_detector": event_detector, "tactical_builder": tactical_builder,
        "brain": brain, "action_executor": action_executor,
    }

    # -----------------------------------------------------------------------
    # PASS 1: CPU
    # -----------------------------------------------------------------------
    result_cpu = run_pass(
        label="CPU", device="cpu",
        capture=capture, hwnd=hwnd,
        shared_components=shared,
        adb_path=ADB_PATH,
        device_serial=DEVICE_SERIAL,
    )

    # Reset tracker/event state between passes for fair comparison
    tracker.__init__()
    event_detector.__init__()

    # -----------------------------------------------------------------------
    # PASS 2: GPU (cuda:0)
    # -----------------------------------------------------------------------
    import torch
    if torch.cuda.is_available():
        result_gpu = run_pass(
            label="GPU (cuda:0)", device="cuda:0",
            capture=capture, hwnd=hwnd,
            shared_components=shared,
            adb_path=ADB_PATH,
            device_serial=DEVICE_SERIAL,
        )
    else:
        print("\n[!] CUDA not available on this machine -- GPU pass SKIPPED.")
        result_gpu = {"label": "GPU (cuda:0)", "device": "cuda:0",
                      "error": "CUDA not available"}

    # -----------------------------------------------------------------------
    # Teardown
    # -----------------------------------------------------------------------
    print("\n[3] Teardown...")
    joystick.release()
    adb.stop()
    capture.close()
    scrcpy_proc.terminate()
    try:
        scrcpy_proc.wait(timeout=3.0)
    except Exception:
        scrcpy_proc.kill()
    print("  Scrcpy terminated, ADB closed, joystick released.")

    # -----------------------------------------------------------------------
    # Apples-to-apples comparison
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  APPLES-TO-APPLES COMPARISON")
    print(SEP)

    cpu_ok  = "error" not in result_cpu
    gpu_ok  = "error" not in result_gpu

    def delta(cpu_v, gpu_v, label, lower_is_better=True):
        if cpu_v is None or gpu_v is None:
            return f"  {label}: CPU=N/A  GPU=N/A"
        diff = gpu_v - cpu_v
        pct_chg = (diff / cpu_v * 100) if cpu_v else 0
        direction = "GPU faster" if diff < 0 else "CPU faster"
        if abs(pct_chg) < 5.0:
            direction = "EQUIVALENT (<5%)"
        return (
            f"  {label:<28}: CPU={cpu_v:.1f}ms  GPU={gpu_v:.1f}ms  "
            f"delta={diff:+.1f}ms ({pct_chg:+.1f}%)  [{direction}]"
        )

    if cpu_ok and gpu_ok:
        print()
        print(delta(result_cpu["total"]["avg"],   result_gpu["total"]["avg"],   "Total tick avg"))
        print(delta(result_cpu["total"]["p50"],   result_gpu["total"]["p50"],   "Total tick p50"))
        print(delta(result_cpu["total"]["p95"],   result_gpu["total"]["p95"],   "Total tick p95"))
        print(delta(result_cpu["vision"]["avg"],  result_gpu["vision"]["avg"],  "YOLO inference avg"))
        print(delta(result_cpu["vision"]["p95"],  result_gpu["vision"]["p95"],  "YOLO inference p95"))
        print(delta(result_cpu["capture"]["avg"], result_gpu["capture"]["avg"], "Capture avg"))
        print(delta(result_cpu["action"]["avg"],  result_gpu["action"]["avg"],  "ActionExecutor avg"))
        print(f"\n  CPU effective FPS : {result_cpu['fps']} FPS")
        print(f"  GPU effective FPS : {result_gpu['fps']} FPS")

        # GPU overhead breakdown
        if result_gpu.get("sync_overhead_ms") is not None:
            print(f"\n  GPU sync overhead (measured):  {result_gpu['sync_overhead_ms']:.2f} ms")
        vision_diff = result_gpu["vision"]["avg"] - result_cpu["vision"]["avg"]
        total_diff  = result_gpu["total"]["avg"]  - result_cpu["total"]["avg"]
        non_vision_diff = total_diff - vision_diff
        print(f"\n  GPU overhead breakdown:")
        print(f"    YOLO-only delta:          {vision_diff:+.2f} ms")
        print(f"    Total tick delta:          {total_diff:+.2f} ms")
        print(f"    Non-YOLO overhead delta:   {non_vision_diff:+.2f} ms  "
              f"(capture + transfer + sync + postprocess)")

        # -----------------------------------------------------------------------
        # PRODUCTION DECISION
        # -----------------------------------------------------------------------
        print(f"\n{SEP}")
        print("  PRODUCTION INFERENCE POLICY DECISION")
        print(SEP)

        cpu_total_avg = result_cpu["total"]["avg"]
        gpu_total_avg = result_gpu["total"]["avg"]
        cpu_p95       = result_cpu["total"]["p95"]
        gpu_p95       = result_gpu["total"]["p95"]
        cpu_fps       = result_cpu["fps"]
        gpu_fps       = result_gpu["fps"]

        total_diff_pct = (gpu_total_avg - cpu_total_avg) / cpu_total_avg * 100

        print(f"""
  Decision criteria (in priority order):
    1. Total tick avg (end-to-end latency):  CPU={cpu_total_avg:.1f}ms  GPU={gpu_total_avg:.1f}ms
    2. Total tick p95 (worst-case):          CPU={cpu_p95:.1f}ms  GPU={gpu_p95:.1f}ms
    3. Effective FPS (no sleep):             CPU={cpu_fps:.1f}  GPU={gpu_fps:.1f}
    4. Stability (lower p95-avg spread):     CPU={cpu_p95-cpu_total_avg:.1f}ms  GPU={gpu_p95-gpu_total_avg:.1f}ms

  Note: isolated YOLO speed does NOT equal lower total tick latency on Pascal (sm_61).
  GPU introduces memory-transfer and scheduling overhead that partially offsets YOLO speedup.
""")

        THRESHOLD_PCT = 5.0  # within 5% = effectively tied

        if abs(total_diff_pct) < THRESHOLD_PCT:
            # Within noise margin -- prefer CPU for simplicity, lower variance
            decision = "cpu"
            reason = (
                f"Total tick difference ({total_diff_pct:+.1f}%) is within the "
                f"{THRESHOLD_PCT}% equivalence threshold. Preferring CPU for simpler "
                f"initialization, zero GPU memory allocation, lower p95 variance, "
                f"and correct behavior on machines without CUDA."
            )
        elif gpu_total_avg < cpu_total_avg:
            decision = "cuda:0"
            reason = (
                f"GPU total tick is {abs(total_diff_pct):.1f}% faster than CPU "
                f"(> {THRESHOLD_PCT}% threshold). GPU is the better production default."
            )
        else:
            decision = "cpu"
            reason = (
                f"CPU total tick is {abs(total_diff_pct):.1f}% faster than GPU. "
                f"GPU overhead on Pascal (sm_61) outweighs YOLO speedup. "
                f"CPU is the correct production default."
            )

        print(f"  DECISION: {decision.upper()}")
        print(f"  REASON:   {reason}")

    else:
        decision = "cpu"  # safe fallback if GPU pass failed
        reason = "GPU pass failed or CUDA unavailable. CPU is the safe fallback."
        print(f"  DECISION: CPU (safe fallback -- GPU pass failed)")

    # -----------------------------------------------------------------------
    # Q-memory post-check
    # -----------------------------------------------------------------------
    print(f"\n{SEP}")
    print("  Q-MEMORY INTEGRITY")
    print(SEP)
    post_q = sha256(Q_BRAIN_PATH)
    post_b = sha256(BASELINE_PATH)
    q_ok = post_q == pre_q
    b_ok = post_b == pre_b
    print(f"  q_brain.json:          {'UNCHANGED [OK]' if q_ok else 'CHANGED [CRITICAL]'}  {post_q[:16]}")
    print(f"  q_brain_baseline.json: {'UNCHANGED [OK]' if b_ok else 'CHANGED [CRITICAL]'}  {post_b[:16]}")
    if not q_ok or not b_ok:
        print("  [CRITICAL] Q-memory was modified during benchmark!")

    # -----------------------------------------------------------------------
    # Save JSON result
    # -----------------------------------------------------------------------
    out = {
        "stage": "27",
        "device_serial": DEVICE_SERIAL,
        "warm_ticks": WARM_TICKS,
        "cpu":      result_cpu,
        "gpu":      result_gpu,
        "decision": decision,
        "reason":   reason,
        "q_brain_unchanged":          q_ok,
        "q_brain_baseline_unchanged": b_ok,
    }
    out_path = os.path.join(WORKSPACE_ROOT, "tools", "stage27_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")
    print(f"\n{SEP}")
    print("  STAGE 27 BENCHMARK COMPLETE")
    print(SEP)

    return decision, result_cpu, result_gpu


if __name__ == "__main__":
    main()
