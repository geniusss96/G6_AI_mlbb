"""
tools/update_readme_p.py -- Stage 27 README P-section updater.
Run once after Stage 27 benchmark to update performance section.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

readme = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
content = open(readme, encoding="utf-8").read()

p_start = content.find("## P. Performance Baseline")
q_start = content.find("## Q. Hotkeys")

new_section = """\
## P. Performance Baseline (Environment-Specific)

Hardware: Samsung Galaxy S22 Ultra SM-S908N (Android 16), Windows PC, NVIDIA P106-100 (sm_61), PyTorch 2.7.1+cu118.

### Stage 27 -- GPU Inference Validated (cuda:0)

| Metric | GPU (cuda:0) |
|---|---|
| YOLO warm avg | ~26.5 ms |
| **Total tick avg** | **~38.8 ms** |
| Total tick p95 | ~42.2 ms |
| Effective FPS | **25.8 FPS** |
| Cold YOLO (JIT) | ~695 ms (one-time) |
| CUDA sync overhead | 0.04 ms (negligible) |

**Production default: `cuda:0`** (the only viable fast-inference path on this hardware).

**Stage 27 Finding -- Ultralytics CPU inference is not viable with GPU visible:**
On this machine (PyTorch 2.7.1+cu118, CUDA-capable GPU present), `model.predict(device='cpu')`
produces ~12000ms per tick instead of the expected ~27ms due to implicit GPU synchronization
inside Ultralytics. This occurs even with `torch.cuda.is_initialized() == False`.
Setting `CUDA_VISIBLE_DEVICES=''` does not hide the GPU on Windows PowerShell.

Consequence: `MLBB_YOLO_DEVICE=cpu` is unsupported on this hardware configuration.
Use `MLBB_YOLO_DEVICE=cpu` only on machines with no CUDA-capable GPU installed.

| Mode | Warm YOLO | Usable |
|---|---|---|
| `cuda:0` | ~26.5 ms | Yes -- production default |
| `cpu` (no GPU present) | ~27 ms | Yes -- GPU-less machines only |
| `cpu` (GPU visible) | ~12000 ms | No -- Ultralytics GPU sync artifact |

### Stage 24 -- Historical CPU Baseline (GPU was NOT yet enabled)

| Metric | Value |
|---|---|
| Warm total tick | ~42.3 ms (p95: 47.0 ms) |
| Warm YOLO inference | ~28.8 ms |
| Effective FPS | ~23.6 FPS |
| Cold YOLO (first inference) | ~1053 ms (JIT warmup, one-time) |

> Stage 24 used `YOLO_DEVICE='0'` (ambiguous). Ultralytics may have resolved this to GPU or CPU depending on version.
> Stage 26/27 switched to the explicit `cuda:0` string. See those stages for authoritative GPU measurements.

Primary bottleneck: **YOLO inference (64-69% of tick)** -- hardware-bound.

"""

new_content = content[:p_start] + new_section + content[q_start:]
open(readme, "w", encoding="utf-8").write(new_content)
print("README P section updated.")
