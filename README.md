# Claude Tactical AI — V2

Autonomous tactical AI for **Mobile Legends: Bang Bang (MLBB)**, controlling the hero **Claude** in real time.

The system uses YOLOv8 computer vision, optical HP/skill-state sensors, a Q-Learning tactical brain, and a modular V2 control stack communicating with an Android device via ADB.

---

## A. Project Purpose

Claude Tactical AI observes the game screen captured from a physical Android device via Scrcpy, runs neural-network object detection and optical sensing, builds a world model, makes tactical decisions through a reinforcement-learning brain, and dispatches joystick/skill/attack inputs to the device through ADB — all in a real-time loop targeting ≥ 5 FPS on tested hardware.

---

## B. V2 Architecture

```
Scrcpy (Android screen mirror)
  ↓
ScreenCapture  (vision/capture.py)
  ↓
YoloDetector + HPDetector + SkillStateChecker  (vision/)
  ↓
WorldTracker → WorldStateBuilder  (world/)
  ↓
WorldEventDetector  (world/events.py)
  ↓
TacticalStateBuilder  (brain/state.py)
  ↓
ClaudeBrainV2  (brain/brain.py)  ←─── Q-Learning (brain/q_learning.py)
  ↓
Action  (actions/models.py)
  ↓
ActionExecutor  (actions/executor.py)
  ↓
JoystickControl / AttackControl / SkillsControl  (control/)
  ↓
ADBTransport  (control/adb.py)
  ↓
Android device (USB / ADB shell)

Learning side-channel:
  WorldEventDetector
    ↓ events
  RewardCalculator  (brain/rewards.py)
    ↓
  LearningIntegrator  (brain/learning_integration.py)
    ↓
  LearningLoop  (brain/learning_loop.py)
    ↓
  QLearningCore  (brain/q_learning.py)
    ↓
  BrainMemory / ExperienceReplay  (brain/memory.py, brain/replay.py)

Orchestration:
  V2Runtime  (app/runtime.py)  ← canonical V2 entry point
```

### Layer Responsibilities (strict boundaries)

| Layer | Responsibility | Forbidden |
|---|---|---|
| `vision/` | Observations only (frames → detections) | No world state, no brain, no ADB |
| `world/` | State construction, tracking, event detection | No control, no brain |
| `brain/` | Tactical reasoning, Q-learning | No ADB, no subprocess, no hardware |
| `actions/` | Action models + execution dispatch | No direct ADB, no tactical logic |
| `control/` | Hardware control via ADBTransport | No tactical decisions |
| `app/runtime.py` | Orchestration only | No embedded tactical logic |

---

## C. Repository Structure

```
скрипт(mlbb)/
│
├── app/                    V2 runtime orchestrator
│   └── runtime.py          ← V2Runtime — canonical entry point
│
├── actions/                Action models and execution dispatch
│   ├── models.py           Action, ActionType
│   └── executor.py         ActionExecutor
│
├── brain/                  Q-Learning tactical brain (V2)
│   ├── brain.py            ClaudeBrainV2
│   ├── q_learning.py       QLearningCore
│   ├── rewards.py          RewardCalculator
│   ├── state.py            TacticalStateBuilder, TacticalState
│   ├── learning_integration.py  LearningIntegrator
│   ├── learning_loop.py    LearningLoop
│   ├── memory.py           BrainMemory
│   ├── replay.py           ExperienceReplay
│   ├── encoder.py          TacticalStateEncoder
│   ├── transition.py       Transition model
│   ├── transition_builder.py
│   ├── action_selection.py ActionSelector (ε-greedy)
│   ├── exploration.py      EpsilonGreedyPolicy
│   ├── tactical.py         TacticalBrain
│   └── v1_adapter.py       V1BrainAdapter (compatibility layer)
│
├── control/                Hardware control layer
│   ├── adb.py              ADBTransport (persistent shell, <0.1 ms dispatch)
│   ├── joystick.py         JoystickControl
│   ├── attack.py           AttackControl
│   ├── skills.py           SkillsControl
│   ├── humanizer.py        InputHumanizer (spatial jitter + hold timing)
│   └── interfaces.py       Port interfaces
│
├── vision/                 Computer vision layer
│   ├── capture.py          ScreenCapture (mss + win32gui)
│   ├── detector.py         YoloDetector (YOLOv8)
│   ├── geometry.py         Coordinate system (canonical 1544×720)
│   ├── hp_detector.py      HPDetector (optical bar sensing)
│   └── skill_state.py      SkillStateChecker (HSV cooldown sensing)
│
├── world/                  World model layer
│   ├── models.py           WorldState, EntityState, Vector2
│   ├── builder.py          WorldStateBuilder
│   ├── tracker.py          WorldTracker (entity identity/lifecycle)
│   └── events.py           WorldEventDetector + WatchdogManager
│
├── config/
│   └── config.py           Centralised configuration (project-relative paths)
│
├── models/
│   └── yolo/v4/best.pt     ← Canonical YOLOv8 model (do not delete)
│
├── data/
│   ├── q_brain_baseline.json  Q-memory integrity baseline
│   └── ...
│
├── q_brain.json            Production Q-learning memory (do not modify manually)
│
├── tests/                  Full offline test suite (264 tests)
│   ├── e2e/                End-to-end harness (Stage 22)
│   ├── brain_regression/   V1/V2 decision parity runner
│   └── test_*.py
│
├── tools/
│   ├── scrcpy/             Bundled Scrcpy 4.1 (Windows x64) ← canonical
│   │   ├── scrcpy.exe
│   │   ├── adb.exe         ← canonical V2 ADB binary
│   │   ├── scrcpy-server
│   │   └── SDL3.dll, AdbWinApi.dll, AdbWinUsbApi.dll, ...
│   ├── real_device_smoke.py   Stage 23B controlled real-device validation
│   └── perf_benchmark.py      Stage 24 performance baseline tool
│
├── legacy/                 Frozen historical implementations (do not use)
│   ├── claude_cv_bot.py
│   ├── combo_macro.py
│   ├── mlbb_assistant.py
│   ├── target_attacker.py
│   └── humanizer.py
│
├── realtime_vision.py      V1 monolithic loop (frozen, compatibility reference)
├── claude_brain.py         V1 brain (used by V1/V2 regression tests)
├── claude_macro.py         V1 macro controller (frozen)
├── joystick_controller.py  V1 joystick (frozen)
│
├── README.md
├── AGENTS.md               Development rules and architecture constraints
├── requirements.txt
└── .gitignore
```

---

## D. Installation

### Requirements

- Python 3.10+
- Windows 10/11 (64-bit) — capture uses `win32gui` / `mss`
- Android device with **USB Debugging** enabled

### Install Python dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt` installs: `opencv-python`, `numpy`, `ultralytics` (YOLOv8), `mss`, `pywin32`, `pynput`, `PyGetWindow`, `tqdm`.

---

## E. Python / Environment Requirements

- Python 3.10 or newer
- `torch` / `torchvision` compatible with your hardware (CPU or CUDA)
- No system-level `adb` required — bundled `tools/scrcpy/adb.exe` is used

> **Note on GPU inference:** `torch.cuda.is_available() = True` does NOT guarantee YOLO runs on GPU. Verify the actual inference device after loading the model. On tested hardware (NVIDIA P106-100), YOLO ran on CPU despite CUDA being available.

---

## F. Device Configuration

Edit `config/config.py` or set environment variables:

```python
# Device serial (or set MLBB_DEVICE_SERIAL env var)
DEVICE_SERIAL = "R3CT90BBMTX"

# Canonical ADB binary (project-relative, resolved automatically)
ADB_PATH = os.path.join(SCRCPY_DIR, "adb.exe")   # tools/scrcpy/adb.exe
```

To find your device serial:
```powershell
.\tools\scrcpy\adb.exe devices
```

The canonical coordinate system is **1544 × 720**. All coordinates in config, brain, and control use this reference. Never create local scaling coefficients — use `vision.geometry` exclusively.

---

## G. Bundled Scrcpy 4.1

The project ships **Scrcpy 4.1** in `tools/scrcpy/` — no separate installation required.

```
tools/scrcpy/
├── scrcpy.exe          ← launch this
├── scrcpy-server       ← pushed to device automatically
├── adb.exe             ← canonical V2 ADB binary
├── SDL3.dll
├── AdbWinApi.dll
├── AdbWinUsbApi.dll
└── avcodec-62.dll, avformat-62.dll, avutil-60.dll, ...
```

Launch the screen mirror:
```powershell
.\tools\scrcpy\scrcpy.exe -s R3CT90BBMTX --window-title scrcpy
```

Scrcpy must be running before starting the V2 runtime.

---

## H. Canonical ADB Path

```
tools/scrcpy/adb.exe
```

This is configured in `config/config.py` as `ADB_PATH` and `ADB_BINARY`. All V2 control modules use `ADBTransport` from `control/adb.py` which reads from this config.

Root-level `adb.exe` / `AdbWinApi.dll` / `AdbWinUsbApi.dll` are preserved as V1 compatibility fallbacks. Do not remove them until V1 modules are fully retired.

---

## I. Canonical YOLO Model Path

```
models/yolo/v4/best.pt
```

Configured in `config/config.py` as `WEIGHTS_PATH`. This model detects: heroes, minions, turrets, buffs, minimap, HP bars across 14 entity classes.

- **Inference size:** `imgsz = 480`
- **Confidence thresholds:** hero/minion = 0.40, buff = 0.22, turret = 0.25
- **Precision:** FP16 (`half=True`) when supported

Do NOT delete or replace this file without running the full regression suite.

---

## J. Running Offline Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Expected result: **264 passed, 0 failed** (no device required).

This covers: brain regression (V1/V2 parity), Q-learning, rewards, events, world model, actions, control (mocked ADB), E2E harness (Stage 22), Stage 19A/19B/20/21 tests, Scrcpy bundle verification.

---

## K. V2 Runtime Entry Point

The canonical V2 runtime is `app/runtime.py` → `V2Runtime`.

To run the V2 loop programmatically:

```python
from app.runtime import V2Runtime

runtime = V2Runtime(
    capture=capture,           # ScreenCapture
    detector=detector,         # YoloDetector
    world_builder=world_builder,
    tracker=tracker,
    event_detector=event_detector,
    tactical_state_builder=tactical_builder,
    brain=brain,               # ClaudeBrainV2
    action_executor=action_executor,
    learning_integrator=learning_integrator,
    hp_detector=hp_detector,
    skill_checker=skill_checker,
)

runtime.run_loop(max_ticks=100)
```

All components are dependency-injected. The runtime is a pure orchestrator — it contains no tactical logic, no ADB calls, and no direct hardware access.

---

## L. Controlled Real-Device Smoke Test

```powershell
python tools/real_device_smoke.py
```

This runs a bounded 10-phase validation: Scrcpy launch → window capture → YOLO inference → WorldState → event detection → 3 controlled physical actions (MOVE, ATTACK, CAST_S1) → isolated brain decision → 5-tick bounded runtime → isolated learning verification → emergency stop → Q-memory hash verification.

**Does NOT start autonomous gameplay.** Safe to run with the device at any screen.

---

## M. Q-Memory Location

| File | Purpose |
|---|---|
| `q_brain.json` | Production Q-learning weights (persists across sessions) |
| `data/q_brain_baseline.json` | Integrity baseline for regression protection |

**Never manually edit `q_brain.json` during active training.**

SHA-256 of both files is verified in the smoke test and E2E tests. If either hash changes unexpectedly, a `CRITICAL` assertion is raised.

---

## N. Legacy / V1 Compatibility Files

These files are **frozen** and must not be modified without an explicit migration task:

| File | Status | Required by |
|---|---|---|
| `realtime_vision.py` | V1 monolithic loop | Historical reference |
| `claude_brain.py` | V1 Q-brain | `tests/test_replay.py`, `tests/brain_regression/` |
| `claude_macro.py` | V1 macro controller | Historical reference |
| `joystick_controller.py` | V1 joystick | Historical reference |
| `legacy/*.py` | Frozen historical | Not imported anywhere |

Do not remove these until a dedicated V2 migration task explicitly approves each removal and verifies zero test regressions.

---

## O. Critical Semantic Rules

### LOST ≠ DEAD

```
Player missing from vision ≠ Player is dead
```

Temporary invisibility (bush, off-screen, occlusion) must **never** set `is_player_dead = True`. Only an explicit `PLAYER_DEATH` event or terminal state context may trigger a death classification.

### Enemy Disappearance ≠ Kill

```
Enemy leaving detection range ≠ HERO_KILL event
```

A kill is only credited after confirmed sustained combat contact (≥ 2 seconds of engagement) followed by enemy disappearance. Passive leaving does not trigger a kill reward.

### Optimized Claude Combo

```
S2 → S2 → S1 → Ultimate
```

This is the validated 4-step combo. There is **no fifth S2-back step**. The combo takes ≈ 1210 ms total. This sequence is preserved in `control/skills.py` and `actions/executor.py` and must not be altered without a dedicated task.

---

## P. Performance Baseline (Stage 24 — Environment-Specific)

Measured on: Samsung Galaxy S22 Ultra SM-S908N (Android 16), Windows PC with NVIDIA P106-100, YOLOv8 inference on CPU.

| Metric | Value |
|---|---|
| Warm total tick | ≈ 42.3 ms (p95: 47.0 ms) |
| Warm YOLO inference | ≈ 28.8 ms (p95: 33.3 ms) |
| Capture (mss) | ≈ 9.0 ms |
| Effective FPS (no sleep) | ≈ 23.6 FPS |
| Cold YOLO (first inference) | ≈ 1053 ms (JIT warmup — one-time only) |
| Model load | ≈ 130 ms (one-time) |
| ADB stdin dispatch | < 0.1 ms (fire-and-forget) |

> These values are **observations on the tested hardware**, not universal guarantees. YOLO performance varies significantly by hardware, CUDA availability, and inference device. ADB dispatch latency does not represent full device round-trip time.

Primary bottleneck: **YOLO inference (68% of tick time)** — hardware-bound. Secondary: **screen capture (21%)**.

---

## Q. Hotkeys (V1 Legacy Runtime Only)

These apply to `realtime_vision.py` (frozen V1):

- **[Space] / [F]** — Force full Claude combo (S2 → S2 → S1 → Ult)
- **[Q] / [Esc]** — Safe shutdown (saves Q-memory, releases joystick)
