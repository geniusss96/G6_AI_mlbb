"""
One-Tap Hero Combo Macro Engine
--------------------------------
Executes frame-sensitive multi-step combos (e.g., Flicker -> Ultimate -> Skill combos)
triggered instantly by physical keyboard hotkeys on the host PC.

Integrates:
  - InputHumanizer for spatial & temporal jitter (anti-robotic behavioral heuristics)
  - Chained single-session ADB execution (drastically reduces per-action latency)
  - Non-blocking keyboard hotkey listener (supports both 'pynput' and native msvcrt fallback)
"""

import sys
import time
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional
from humanizer import InputHumanizer


# ==============================================================================
# CONFIGURATION & HERO COMBO PRESETS
# ==============================================================================
@dataclass
class ComboStep:
    name: str
    target_xy: Tuple[int, int]
    base_delay_after_ms: float   # Delay to wait before next step (ms)
    hold_duration_ms: float = 55.0  # Press-and-hold duration (ms)


class ComboConfig:
    # Optional device serial if multiple devices are attached via USB
    DEVICE_SERIAL = ""

    # Screen resolution (Landscape)
    SCREEN_WIDTH = 2400
    SCREEN_HEIGHT = 1080

    # --------------------------------------------------------------------------
    # COMBO PRESET 1: Instant Engage (e.g. Flicker -> Ultimate -> Skill 1 -> BA)
    # Trigger Key: 'F'
    # --------------------------------------------------------------------------
    ENGAGE_COMBO = [
        ComboStep(
            name="Battle Spell (Flicker)",
            target_xy=(1950, 850),
            base_delay_after_ms=30.0,
            hold_duration_ms=50.0
        ),
        ComboStep(
            name="Ultimate Skill",
            target_xy=(2180, 650),
            base_delay_after_ms=45.0,
            hold_duration_ms=65.0
        ),
        ComboStep(
            name="Skill 1 Follow-up",
            target_xy=(1900, 600),
            base_delay_after_ms=30.0,
            hold_duration_ms=50.0
        ),
        ComboStep(
            name="Basic Attack",
            target_xy=(2150, 880),
            base_delay_after_ms=0.0,
            hold_duration_ms=45.0
        ),
    ]

    # --------------------------------------------------------------------------
    # COMBO PRESET 2: Rapid Skill Burst (e.g. Skill 2 -> Skill 1 -> Auto-Attack)
    # Trigger Key: 'R'
    # --------------------------------------------------------------------------
    BURST_COMBO = [
        ComboStep(
            name="Skill 2",
            target_xy=(2050, 720),
            base_delay_after_ms=40.0,
            hold_duration_ms=50.0
        ),
        ComboStep(
            name="Skill 1",
            target_xy=(1900, 600),
            base_delay_after_ms=35.0,
            hold_duration_ms=50.0
        ),
        ComboStep(
            name="Basic Attack",
            target_xy=(2150, 880),
            base_delay_after_ms=0.0,
            hold_duration_ms=40.0
        ),
    ]


# ==============================================================================
# COMBO EXECUTOR
# ==============================================================================
class ComboExecutor:
    """Dispatches humanized combo sequences over ADB with minimal latency."""

    def __init__(self, serial: str = ""):
        self.serial = serial.strip()
        self.base_cmd = ["adb"]
        if self.serial:
            self.base_cmd.extend(["-s", self.serial])

        self._lock = threading.Lock()
        self.is_executing = False

    def execute_combo_async(self, combo_name: str, steps: List[ComboStep]) -> None:
        """Fires the combo sequence in a dedicated background worker."""
        if not self._lock.acquire(blocking=False):
            print(f"[!] Warning: Combo '{combo_name}' dropped (another combo is still active).")
            return

        def _worker():
            try:
                self.is_executing = True
                t_start = time.perf_counter()
                print(f"\n[>>>] Firing Combo: '{combo_name}' ({len(steps)} steps)...")

                commands = []
                for idx, step in enumerate(steps):
                    # 1. Apply spatial Gaussian jitter to target coordinates
                    jx, jy = InputHumanizer.jitter_coordinate(
                        step.target_xy[0],
                        step.target_xy[1],
                        std_dev=3.5,
                        max_offset=7
                    )

                    # 2. Simulate human physical touch-down to touch-up hold window
                    # In Android, `input swipe X Y X Y <duration_ms>` simulates a tap with an exact hold duration!
                    hold_ms = int(round(step.hold_duration_ms + InputHumanizer.get_touch_hold_duration() * 1000 - 65))
                    hold_ms = max(35, min(120, hold_ms))
                    commands.append(f"input swipe {jx} {jy} {jx} {jy} {hold_ms}")

                    # 3. Add inter-step delay with temporal variance (if not last step)
                    if idx < len(steps) - 1:
                        inter_delay_sec = InputHumanizer.get_reaction_delay(
                            base_delay_ms=step.base_delay_after_ms,
                            variance_ms=8.0,
                            min_floor_ms=10.0
                        )
                        inter_delay_us = int(inter_delay_sec * 1_000_000)
                        commands.append(f"usleep {inter_delay_us}")

                # Chain entire combo into single ADB session
                chained_script = " && ".join(commands)
                cmd = self.base_cmd + ["shell", chained_script]

                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                t_total = (time.perf_counter() - t_start) * 1000
                print(f"[+] Combo '{combo_name}' finished in {t_total:.1f}ms (Return code: {proc.returncode})")

            finally:
                self.is_executing = False
                self._lock.release()

        threading.Thread(target=_worker, daemon=True).start()


# ==============================================================================
# KEYBOARD LISTENER & MAIN LOOP
# ==============================================================================
def start_pynput_listener(executor: ComboExecutor):
    """Listens for global physical hotkeys using pynput."""
    from pynput import keyboard

    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char:
                ch = key.char.lower()
                if ch == 'f':
                    executor.execute_combo_async("Instant Engage (Flicker-Ult)", ComboConfig.ENGAGE_COMBO)
                elif ch == 'r':
                    executor.execute_combo_async("Rapid Skill Burst", ComboConfig.BURST_COMBO)
                elif ch == 'q':
                    print("[*] 'q' detected. Stopping listener.")
                    return False
        except Exception as e:
            print(f"[!] Hotkey handler error: {e}")

    print("[*] Global hotkey hook active (pynput).")
    print("    [F] -> Trigger Engage Combo (Flicker + Ult + S1 + Attack)")
    print("    [R] -> Trigger Burst Combo (S2 + S1 + Attack)")
    print("    [Q] -> Exit")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def start_msvcrt_listener(executor: ComboExecutor):
    """Windows native console keyboard listener (fallback if pynput is not installed)."""
    import msvcrt

    print("[*] Console keyboard listener active (msvcrt). Keep this console focused.")
    print("    [F] -> Trigger Engage Combo (Flicker + Ult + S1 + Attack)")
    print("    [R] -> Trigger Burst Combo (S2 + S1 + Attack)")
    print("    [Q] -> Exit")

    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if ch == 'f':
                executor.execute_combo_async("Instant Engage (Flicker-Ult)", ComboConfig.ENGAGE_COMBO)
            elif ch == 'r':
                executor.execute_combo_async("Rapid Skill Burst", ComboConfig.BURST_COMBO)
            elif ch == 'q':
                print("[*] 'q' pressed. Shutting down.")
                break
        time.sleep(0.01)


def main():
    print("==================================================")
    print("      MLBB One-Tap Hero Combo Macro Engine        ")
    print("==================================================")
    executor = ComboExecutor(serial=ComboConfig.DEVICE_SERIAL)

    # Check for pynput availability
    try:
        import pynput
        start_pynput_listener(executor)
    except ImportError:
        print("[!] Note: 'pynput' not installed. Falling back to native console listener.")
        print("[!] Tip: Run 'pip install pynput' for background global hotkeys across all windows.")
        start_msvcrt_listener(executor)


if __name__ == "__main__":
    main()
