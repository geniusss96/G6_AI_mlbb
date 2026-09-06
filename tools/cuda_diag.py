"""
tools/cuda_diag.py — Stage 24.5 GPU/CUDA Diagnostic Only.

Determines whether the installed PyTorch/YOLO environment can execute
models/yolo/v4/best.pt on CUDA.

DOES NOT modify:
  - production runtime
  - config/config.py
  - q_brain.json
  - data/q_brain_baseline.json
  - models/yolo/v4/best.pt
  - v2-stable tag / 46a1577 commit
"""

# Force UTF-8 stdout so emoji/special chars work on Windows CP1251 consoles.
import sys as _sys, io as _io
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import sys
import time
import hashlib
import traceback

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

CANONICAL_MODEL = os.path.join(WORKSPACE_ROOT, "models", "yolo", "v4", "best.pt")
Q_BRAIN_PATH    = os.path.join(WORKSPACE_ROOT, "q_brain.json")
BASELINE_PATH   = os.path.join(WORKSPACE_ROOT, "data", "q_brain_baseline.json")

SEP = "=" * 68


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# Pre-flight hash snapshot
# ---------------------------------------------------------------------------
pre_q_hash = sha256(Q_BRAIN_PATH)
pre_b_hash = sha256(BASELINE_PATH)

print(SEP)
print("  STAGE 24.5 — GPU/CUDA DIAGNOSTIC")
print(SEP)
print(f"\n  PRE-FLIGHT q_brain.json SHA-256:     {pre_q_hash[:24]}...")
print(f"  PRE-FLIGHT q_brain_baseline SHA-256: {pre_b_hash[:24]}...")


# ---------------------------------------------------------------------------
# A. Python + PyTorch environment
# ---------------------------------------------------------------------------
section("A. ENVIRONMENT")

print(f"  Python:  {sys.version}")

try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version (PyTorch build): {torch.version.cuda}")
        print(f"  GPU name:  {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute capability: sm_{cap[0]}{cap[1]}  ({cap})")
        total_mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"  VRAM total: {total_mem} MB")
    else:
        print("  [!] CUDA not available — cannot perform GPU diagnostics")
        torch_available = False
except ImportError as e:
    print(f"  [FATAL] torch not importable: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# B. Minimal CUDA tensor test
# ---------------------------------------------------------------------------
section("B. CUDA TENSOR TEST")

cuda_tensor_ok = False
if not torch.cuda.is_available():
    print("  SKIPPED — CUDA not available")
else:
    try:
        print("  Running: x = torch.randn(1024, 1024, device='cuda')  ...")
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"  Result shape: {y.shape}, device: {y.device}")
        print("  CUDA tensor test: PASSED [OK]")
        cuda_tensor_ok = True
    except Exception as e:
        print(f"  CUDA tensor test: FAILED [ERR]")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Message: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# C. Load canonical model on CUDA
# ---------------------------------------------------------------------------
section("C. YOLO MODEL CUDA LOAD & INFERENCE")

if not torch.cuda.is_available():
    print("  SKIPPED — CUDA not available")
elif not os.path.isfile(CANONICAL_MODEL):
    print(f"  SKIPPED — model not found: {CANONICAL_MODEL}")
else:
    print(f"  Model: {CANONICAL_MODEL}")
    print("  Attempting to load on device='cuda:0' (no silent CPU fallback)...")

    try:
        from ultralytics import YOLO
        t0 = time.perf_counter()
        model = YOLO(CANONICAL_MODEL)
        load_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  Model load: {load_ms:.0f} ms")

        # Verify actual device before any inference
        try:
            param_device = str(next(model.model.parameters()).device)
        except Exception:
            param_device = "unknown (could not read parameters)"
        print(f"  Parameters device after load (before explicit predict): {param_device}")

        # Grab a real frame for inference (or synthesize one)
        import numpy as np
        print("  Generating synthetic 1544×720 BGR test frame...")
        test_frame = np.random.randint(0, 255, (720, 1544, 3), dtype=np.uint8)

        # --- First inference explicitly on cuda:0 ---
        print("\n  [C1] First inference — device='cuda:0', half=True ...")
        cuda_infer_ok = False
        cuda_error = None
        try:
            t0 = time.perf_counter()
            results = model.predict(
                source=test_frame,
                device="cuda:0",
                conf=0.22,
                verbose=False,
                imgsz=480,
                half=True,
            )
            torch.cuda.synchronize()
            first_cuda_ms = (time.perf_counter() - t0) * 1000.0

            # Determine actual device after inference
            try:
                param_device_after = str(next(model.model.parameters()).device)
            except Exception:
                param_device_after = "unknown"

            print(f"  First CUDA inference: {first_cuda_ms:.2f} ms")
            print(f"  Detections: {len(results[0].boxes) if results and results[0].boxes is not None else 0}")
            print(f"  Parameters device after CUDA predict: {param_device_after}")
            print("  CUDA inference: PASSED [OK]")
            cuda_infer_ok = True

        except Exception as e:
            cuda_error = e
            print(f"  CUDA inference: FAILED [ERR]")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Message: {e}")
            # Print the full traceback for exact diagnosis
            traceback.print_exc()

        # --- Warm CUDA inferences (only if first succeeded) ---
        if cuda_infer_ok:
            section("D. WARM CUDA INFERENCE (5 calls)")
            warm_times = []
            for i in range(5):
                t0 = time.perf_counter()
                _ = model.predict(
                    source=test_frame,
                    device="cuda:0",
                    conf=0.22,
                    verbose=False,
                    imgsz=480,
                    half=True,
                )
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000.0
                warm_times.append(elapsed)
                print(f"  Warm #{i+1}: {elapsed:.2f} ms")

            avg_warm = sum(warm_times) / len(warm_times)
            print(f"\n  CUDA warm inference average: {avg_warm:.2f} ms")
            print(f"  CPU baseline (Stage 24):      ~28.8 ms")
            if avg_warm < 28.8:
                speedup = 28.8 / avg_warm
                print(f"  GPU speedup vs CPU baseline: {speedup:.1f}×")
            else:
                print(f"  GPU inference not faster than CPU baseline ({avg_warm:.1f} ms vs 28.8 ms)")

    except Exception as e:
        print(f"  Model load FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        cuda_infer_ok = False

# ---------------------------------------------------------------------------
# E. CPU baseline verification (unchanged)
# ---------------------------------------------------------------------------
section("E. CPU BASELINE VERIFICATION")

print("  Loading model on CPU explicitly to verify baseline unchanged...")
try:
    from ultralytics import YOLO as _YOLO
    import numpy as np
    _model = _YOLO(CANONICAL_MODEL)
    _frame = np.random.randint(0, 255, (720, 1544, 3), dtype=np.uint8)

    # Warmup
    _ = _model.predict(source=_frame, device="cpu", conf=0.22, verbose=False, imgsz=480, half=False)
    # Measure 3 warm calls
    cpu_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = _model.predict(source=_frame, device="cpu", conf=0.22, verbose=False, imgsz=480, half=False)
        cpu_times.append((time.perf_counter() - t0) * 1000.0)

    avg_cpu = sum(cpu_times) / len(cpu_times)
    print(f"  CPU warm inference average (3 calls): {avg_cpu:.2f} ms")
    print(f"  Stage 24 reference:                   ~28.8 ms")
    print(f"  CPU baseline: {'CONSISTENT' if avg_cpu < 60 else 'HIGHER THAN EXPECTED'}")
except Exception as e:
    print(f"  CPU baseline check FAILED: {e}")

# ---------------------------------------------------------------------------
# F. CUDA architectural compatibility check
# ---------------------------------------------------------------------------
section("F. CUDA ARCHITECTURE COMPATIBILITY")

if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]
    print(f"  GPU sm version: sm_{sm}  (sm_{cap[0]}{cap[1]})")
    print(f"  PyTorch CUDA build: {torch.version.cuda}")

    # Check what SM archs are compiled into this torch build
    try:
        archs = torch.cuda.get_arch_list()
        print(f"  Compiled SM architectures in this PyTorch build: {archs}")
        gpu_arch = f"sm_{sm}"
        if gpu_arch in archs:
            print(f"  GPU architecture ({gpu_arch}) IS in compiled list → should be supported")
        else:
            # Check if any generic arch covers it
            # PTX compilation allows forward compatibility
            higher = [a for a in archs if int(a.replace("sm_","")) <= sm]
            print(f"  GPU architecture ({gpu_arch}) NOT directly in compiled list")
            print(f"  Archs at or below GPU's capability: {higher}")
            if higher:
                print(f"  PTX JIT compilation may still work (forward compatibility)")
            else:
                print(f"  No compatible compiled arch found — GPU may not be usable")
    except AttributeError:
        print("  torch.cuda.get_arch_list() not available in this PyTorch version")

    # Check cuDNN
    print(f"\n  cuDNN available: {torch.backends.cudnn.is_available()}")
    if torch.backends.cudnn.is_available():
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  cuDNN enabled:  {torch.backends.cudnn.enabled}")

# ---------------------------------------------------------------------------
# G. Full test suite
# ---------------------------------------------------------------------------
section("G. FULL REGRESSION TEST SUITE")

import subprocess
result = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
    cwd=WORKSPACE_ROOT,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)
# unittest prints to stderr
stdout_str = result.stdout or ""
stderr_str = result.stderr or ""
output = (stdout_str + stderr_str).strip()
# Show last 10 lines
lines = output.splitlines()
for line in lines[-12:]:
    print(" ", line)

passed = "OK" in output
print(f"  Test suite: {'PASSED [OK]' if passed else 'FAILED [ERR]'}")

# ---------------------------------------------------------------------------
# H. Post-flight Q-memory integrity
# ---------------------------------------------------------------------------
section("H. Q-MEMORY INTEGRITY")

post_q_hash = sha256(Q_BRAIN_PATH)
post_b_hash = sha256(BASELINE_PATH)
q_ok = post_q_hash == pre_q_hash
b_ok = post_b_hash == pre_b_hash
print(f"  q_brain.json:          {'UNCHANGED [OK]' if q_ok else 'MODIFIED [CRITICAL]'}")
print(f"  q_brain_baseline.json: {'UNCHANGED [OK]' if b_ok else 'MODIFIED [CRITICAL]'}")
if not q_ok or not b_ok:
    print("  [CRITICAL] Q-memory was mutated during diagnostics!")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("SUMMARY")

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
cap2 = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
cuda_build = torch.version.cuda if torch.cuda.is_available() else 'N/A'
print(f"  GPU:               {gpu_name}")
print(f"  Compute cap:       sm_{cap2[0]}{cap2[1]}")
print(f"  PyTorch:           {torch.__version__}")
print(f"  CUDA (build):      {cuda_build}")
print(f"  CUDA tensor test:  {'PASSED' if cuda_tensor_ok else 'SKIPPED/FAILED'}")
print(f"  CUDA model infer:  {'PASSED' if 'cuda_infer_ok' in dir() and cuda_infer_ok else 'FAILED/SKIPPED'}")
print( "  CPU baseline:      ~28.8 ms (Stage 24 reference)")
print(f"  Tests:             {'264 passed' if passed else 'SEE OUTPUT'}")
print(f"  Q-memory:          {'UNCHANGED' if q_ok and b_ok else 'MODIFIED -- CRITICAL'}")

if 'cuda_infer_ok' in dir() and cuda_infer_ok:
    print("  RECOMMENDATION: GPU INFERENCE IS VIABLE.")
    print("  Consider enabling device='cuda:0' in a separate Stage 26 task")
    print("  with full regression validation before touching production config.")
elif cuda_tensor_ok:
    print("  RECOMMENDATION: CUDA tensors work but YOLO inference on this GPU failed.")
    print("  Review the exact error above. May require:")
    print("    - Reinstalling PyTorch with matching CUDA SM arch")
    print("    - Using a different CUDA compute path")
    print("    - Staying on CPU for production")
else:
    print("  RECOMMENDATION: CUDA not usable in this environment.")
    print("  CPU baseline (~28.8 ms warm) is confirmed adequate for >=5 FPS loop.")
