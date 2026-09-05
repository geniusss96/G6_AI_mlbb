"""
Mobile Legends: Bang Bang - Claude (Клод) Production-Grade Combo Macro
========================================================================
Engineered with Python, pynput, and optimized ADB subprocess execution.

Hero Mechanics (Claude):
  - Engage & Burst Combo [F]:
      Places Skill 2 Shadow -> Flickers into enemy backline ->
      Instantly activates Ultimate (Blazing Duet) -> Fires Skill 1 (Art of Thievery)
      to steal enemy attack & movement speed, accelerating Ultimate barrage DPS.
  - Reposition & Escape [R]:
      Reactivates Skill 2 to swap back to Dexter's shadow -> Instant Basic Attack.

Features:
  - Self-contained InputHumanizer with Gaussian spatial jitter (mu=0, sigma=3.0px)
  - Realistic touch-down hold windows (45ms to 95ms) via `input swipe`
  - Temporal jitter for inter-skill delays using a normal distribution
  - SWAP_XY coordinate translation toggle for portrait-to-landscape digitizer alignment
  - Chained subshell execution (`&&` and `usleep`) for single-session sub-frame dispatch
  - Automatic ADB binary lookup (local folder or system PATH)
  - Dual hotkey engine: global pynput hooks with robust msvcrt console fallback
"""

import os
import sys
import time
import random
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Tuple, List, Optional


# ==============================================================================
# SELF-CONTAINED INPUT HUMANIZATION & ANTI-DETECTION ENGINE
# ==============================================================================
class InputHumanizer:
    """Simulates realistic human physical touch biomechanics and reaction timing."""

    @staticmethod
    def jitter_coordinate(
        x: int,
        y: int,
        std_dev: float = 3.0,
        max_offset: int = 6
    ) -> Tuple[int, int]:
        """
        Applies 2D Gaussian jitter to target coordinates.
        Prevents static pixel detection heuristics by clustering taps naturally.
        """
        dx = random.gauss(0, std_dev)
        dy = random.gauss(0, std_dev)
        dx = max(-max_offset, min(max_offset, dx))
        dy = max(-max_offset, min(max_offset, dy))
        return int(round(x + dx)), int(round(y + dy))

    @staticmethod
    def get_touch_hold_duration(
        mean_ms: float = 65.0,
        std_ms: float = 10.0,
        min_ms: float = 45.0,
        max_ms: float = 95.0
    ) -> int:
        """
        Calculates physical glass contact duration in milliseconds (45ms to 95ms).
        Human fingers physically compress capacitive glass for ~50-90ms.
        """
        val = random.gauss(mean_ms, std_ms)
        clamped = max(min_ms, min(max_ms, val))
        return int(round(clamped))

    @staticmethod
    def get_reaction_delay(
        base_ms: float = 30.0,
        variance_ms: float = 5.0,
        min_floor_ms: float = 8.0,
        **kwargs
    ) -> int:
        """
        Calculates inter-skill delay with Gaussian variance to avoid rigid clock ticks.
        Returns delay in milliseconds. Safely handles base_delay_ms alias.
        """
        if "base_delay_ms" in kwargs:
            base_ms = kwargs["base_delay_ms"]
        delay = random.gauss(base_ms, variance_ms)
        clamped = max(min_floor_ms, delay)
        return int(round(clamped))


from config.config import (
    DEVICE_SERIAL,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SWAP_XY,
    COORD_SKILL_1,
    COORD_SKILL_2,
    COORD_ULTIMATE,
    COORD_FLICKER,
    COORD_BASIC_ATTACK,
    SPATIAL_JITTER_SIGMA,
    SPATIAL_MAX_OFFSET,
    HOLD_DURATION_MIN_MS,
    HOLD_DURATION_MAX_MS,
    HOLD_DURATION_MEAN_MS,
)


# ==============================================================================
# CLAUDE CONFIGURATION & CALIBRATED COORDINATES
# ==============================================================================
class ClaudeConfig:
    # --------------------------------------------------------------------------
    # Device Identification & Resolution
    # --------------------------------------------------------------------------
    DEVICE_SERIAL = DEVICE_SERIAL
    SCREEN_WIDTH = SCREEN_WIDTH
    SCREEN_HEIGHT = SCREEN_HEIGHT

    # --------------------------------------------------------------------------
    # Orientation & Coordinate Space Translation Toggle
    # --------------------------------------------------------------------------
    SWAP_XY = SWAP_XY

    # --------------------------------------------------------------------------
    # Claude HUD Button Coordinates (User's Exact Calibrated Values)
    # --------------------------------------------------------------------------
    COORD_SKILL_1 = COORD_SKILL_1
    COORD_SKILL_2 = COORD_SKILL_2
    COORD_ULTIMATE = COORD_ULTIMATE
    COORD_FLICKER = COORD_FLICKER
    COORD_BASIC_ATTACK = COORD_BASIC_ATTACK

    # --------------------------------------------------------------------------
    # Anti-Detection Jitter Parameters
    # --------------------------------------------------------------------------
    SPATIAL_JITTER_SIGMA = SPATIAL_JITTER_SIGMA
    SPATIAL_MAX_OFFSET = SPATIAL_MAX_OFFSET
    HOLD_DURATION_MIN_MS = HOLD_DURATION_MIN_MS
    HOLD_DURATION_MAX_MS = HOLD_DURATION_MAX_MS
    HOLD_DURATION_MEAN_MS = HOLD_DURATION_MEAN_MS


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class SkillAction:
    name: str
    coords: Tuple[int, int]
    delay_after_ms: float         # Base delay before next action in sequence (ms)
    hold_duration_ms: Optional[float] = None


# ==============================================================================
# LOW-LATENCY ADB COMBO CONTROLLER
# ==============================================================================
class ClaudeMacroController:
    """Manages low-latency ADB connection, coordinate translation, and chained execution."""

    def __init__(self, serial: str = ""):
        self.serial = serial.strip()
        self.adb_path = self._find_adb_binary()
        self.base_cmd = [self.adb_path]
        if self.serial:
            self.base_cmd.extend(["-s", self.serial])

        self._lock = threading.Lock()
        self.joystick = None
        self._verify_connection()

    def set_joystick(self, joystick) -> None:
        """Подключает экземпляр JoystickController для отправки тапов через общий ADB shell."""
        self.joystick = joystick

    def _send_fast_cmd(self, cmd_line: str) -> None:
        """
        Отправляет команду тапа/свайпа через открытый интерактивный ADB Shell джойстика
        (задержка <0.1мс вместо 100мс у subprocess.Popen) и уведомляет джойстик о касании экрана.
        """
        if self.joystick:
            self.joystick.send_raw_cmd(cmd_line)
            self.joystick.notify_screen_touched()
        else:
            cmd = self.base_cmd + ["shell", cmd_line]
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    @staticmethod
    def _find_adb_binary() -> str:
        """Locates adb.exe in local directories, platform-tools, or system PATH."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "adb.exe"),
            os.path.join(script_dir, "platform-tools", "adb.exe"),
            os.path.join(script_dir, "adb"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

        which_adb = shutil.which("adb")
        if which_adb:
            return which_adb

        return "adb"

    def _verify_connection(self) -> None:
        """Checks if ADB daemon can see the target device."""
        try:
            res = subprocess.run(
                self.base_cmd + ["get-state"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            state = res.stdout.strip()
            if state != "device":
                raise ConnectionError(f"Device state is '{state}', expected 'device'.")
            print(f"[+] ADB Binary: {self.adb_path}")
            print(f"[+] Device connected successfully. Status: {state}")
        except Exception as e:
            print(f"[!] ADB Error: Could not connect to device.")
            print(f"[!] Ensure USB Debugging is ON and phone is connected. ({e})")
            sys.exit(1)

    @staticmethod
    def transform_coords(raw_x: int, raw_y: int) -> Tuple[int, int]:
        """
        Translates raw digitizer coordinates into exact landscape input coordinates (720x1544, rotation 3):
        - Landscape X = raw_y (long axis: 1000..1400 -> right side)
        - Landscape Y = 720 - raw_x (short axis inverted -> bottom corner, avoiding top UI)
        """
        if ClaudeConfig.SWAP_XY:
            tx = raw_y
            ty = ClaudeConfig.SCREEN_HEIGHT - raw_x
            return tx, ty
        return raw_x, raw_y

    def execute_sequence_async(self, combo_name: str, actions: List[SkillAction]) -> None:
        """Dispatches the chained combo sequence in a non-blocking background worker."""
        if not self._lock.acquire(blocking=False):
            print(f"[!] Dropped trigger for '{combo_name}' (another combo is still active).")
            return

        def _worker():
            try:
                t_start = time.perf_counter()
                print(f"\n[>>>] Firing Claude Combo: {combo_name} ({len(actions)} steps)...")

                commands = []
                for i, action in enumerate(actions):
                    # 1. Translate coordinates according to orientation toggle (SWAP_XY)
                    raw_x, raw_y = action.coords
                    tx, ty = self.transform_coords(raw_x, raw_y)

                    # 2. Apply spatial Gaussian jitter (mu=0, sigma=3.0) on target axes
                    jx, jy = InputHumanizer.jitter_coordinate(
                        tx,
                        ty,
                        std_dev=ClaudeConfig.SPATIAL_JITTER_SIGMA,
                        max_offset=ClaudeConfig.SPATIAL_MAX_OFFSET
                    )

                    # 3. Calculate realistic human touch-down duration
                    if action.hold_duration_ms:
                        hold_ms = int(round(action.hold_duration_ms))
                    else:
                        hold_ms = InputHumanizer.get_touch_hold_duration(
                            mean_ms=ClaudeConfig.HOLD_DURATION_MEAN_MS,
                            std_ms=10.0,
                            min_ms=ClaudeConfig.HOLD_DURATION_MIN_MS,
                            max_ms=ClaudeConfig.HOLD_DURATION_MAX_MS
                        )

                    commands.append(f"input swipe {jx} {jy} {jx} {jy} {hold_ms}")

                    # 4. Add inter-action delay with temporal variance
                    if i < len(actions) - 1:
                        raw_delay = action.delay_after_ms
                        jittered_delay = InputHumanizer.get_reaction_delay(
                            base_ms=raw_delay,
                            variance_ms=max(1.0, raw_delay * 0.08),
                            min_floor_ms=5.0
                        )
                        delay_us = int(jittered_delay * 1000)
                        commands.append(f"usleep {delay_us}")

                # Chain commands with &&
                chained_shell_cmd = " && ".join(commands)
                cmd = self.base_cmd + ["shell", chained_shell_cmd]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                t_elapsed = (time.perf_counter() - t_start) * 1000
                print(f"[+] '{combo_name}' executed in {t_elapsed:.1f}ms (Exit code: {result.returncode})")

                if self.joystick:
                    self.joystick.notify_screen_touched()

            except subprocess.TimeoutExpired:
                print(f"[!] Warning: Combo '{combo_name}' timed out after 5.0 seconds.")
            except Exception as e:
                print(f"[!] Error executing combo sequence: {e}")
            finally:
                self._lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    # --------------------------------------------------------------------------
    # Claude Preset Combos
    # --------------------------------------------------------------------------
    def trigger_engage_ultimate(self, ult_channel_ms: float = 1000.0) -> None:
        """
        Истинное комбо Клода (Dive & Burst):
        1. Скилл 2 (Тень): оставляет Декстера (120мс вылет)
        2. Скилл 2 (Телепорт): прыгает во врагов
        3. Скилл 1 (Воровство): крадет скорость атаки для разгона ульты
        4. Ультимейт (Буйство стрельбы): вращается и наносит взрывной урон (~1.0с)
        """
        actions = [
            SkillAction(
                name="Skill 2 (Place Dexter)",
                coords=ClaudeConfig.COORD_SKILL_2,
                delay_after_ms=120.0,
                hold_duration_ms=45.0
            ),
            SkillAction(
                name="Skill 2 (Teleport In)",
                coords=ClaudeConfig.COORD_SKILL_2,
                delay_after_ms=50.0,
                hold_duration_ms=40.0
            ),
            SkillAction(
                name="Skill 1 (Art of Thievery)",
                coords=ClaudeConfig.COORD_SKILL_1,
                delay_after_ms=40.0,
                hold_duration_ms=40.0
            ),
            SkillAction(
                name="Ultimate (Blazing Duet)",
                coords=ClaudeConfig.COORD_ULTIMATE,
                delay_after_ms=ult_channel_ms,
                hold_duration_ms=55.0
            ),
        ]
        self.execute_sequence_async("CLAUDE TRUE COMBO (S2 -> S2 -> S1 -> ULT)", actions)

    def fire_basic_attack_fast(self) -> None:
        """
        Мгновенный высокоскоростной выстрел авто-атаки (0.1мс через открытый shell).
        Бьет точно в центр кнопки атаки (1350, 560) с рандомизацией античита.
        """
        jx, jy = InputHumanizer.jitter_coordinate(1350, 560, std_dev=2.0, max_offset=4)
        hold_ms = InputHumanizer.get_touch_hold_duration(mean_ms=45.0, std_ms=6.0, min_ms=30.0, max_ms=65.0)
        self._send_fast_cmd(f"input swipe {jx} {jy} {jx} {jy} {hold_ms}")

    def fire_skill_1_fast(self) -> None:
        """
        Мгновенный каст 1-го скилла (Искусство воровства: 1088, 540).
        Крадет скорость атаки и движения у врагов для разгона урона.
        """
        jx, jy = InputHumanizer.jitter_coordinate(1088, 540, std_dev=2.0, max_offset=4)
        hold_ms = InputHumanizer.get_touch_hold_duration(mean_ms=45.0, std_ms=6.0, min_ms=30.0, max_ms=65.0)
        self._send_fast_cmd(f"input swipe {jx} {jy} {jx} {jy} {hold_ms}")

    def fire_skill_2_fast(self) -> None:
        """Мгновенный каст 2-го скилла (Зеркальное отражение / Свап)."""
        jx, jy = InputHumanizer.jitter_coordinate(1228, 392, std_dev=2.0, max_offset=4)
        hold_ms = InputHumanizer.get_touch_hold_duration(mean_ms=45.0, std_ms=6.0, min_ms=30.0, max_ms=65.0)
        self._send_fast_cmd(f"input swipe {jx} {jy} {jx} {jy} {hold_ms}")

    def panic_retreat_s2(self) -> None:
        """
        Панический отскок на С2: бросает тень назад к вышке и мгновенно свапается, спасая жизнь.
        """
        actions = [
            SkillAction(
                name="Panic Skill 2 (Place Shadow)",
                coords=ClaudeConfig.COORD_SKILL_2,
                delay_after_ms=120.0,
                hold_duration_ms=45.0
            ),
            SkillAction(
                name="Panic Skill 2 (Swap to Safety)",
                coords=ClaudeConfig.COORD_SKILL_2,
                delay_after_ms=0.0,
                hold_duration_ms=45.0
            ),
        ]
        self.execute_sequence_async("PANIC RETREAT (S2 Double Tap)", actions)

    def trigger_regen(self) -> None:
        """Прожатие кнопки 'Восстановление' (Хил/Реген) в безопасный момент."""
        jx, jy = InputHumanizer.jitter_coordinate(815, 640, std_dev=2.0, max_offset=4)
        hold_ms = InputHumanizer.get_touch_hold_duration(mean_ms=50.0, std_ms=8.0, min_ms=35.0, max_ms=75.0)
        self._send_fast_cmd(f"input swipe {jx} {jy} {jx} {jy} {hold_ms}")

    def level_up_skills(self) -> None:
        """
        Автоматическая прокачка способностей: быстро кликает по плюсикам над скиллами
        Приоритет: Ульта (1051, 305) -> Скилл 1 (1088, 465) -> Скилл 2 (1228, 320).
        """
        plus_buttons = [(1051, 305), (1088, 465), (1228, 320)]
        for px, py in plus_buttons:
            jx, jy = InputHumanizer.jitter_coordinate(px, py, std_dev=2.0, max_offset=3)
            self._send_fast_cmd(f"input tap {jx} {jy}")

    def quick_buy_item(self) -> None:
        """
        Быстрая покупка рекомендованного предмета в правом верхнем углу (1220, 160).
        """
        jx, jy = InputHumanizer.jitter_coordinate(1220, 160, std_dev=3.0, max_offset=6)
        self._send_fast_cmd(f"input tap {jx} {jy}")

    def trigger_basic_attack(self) -> None:
        """Одиночный авто-удар по кнопке атаки."""
        self.fire_basic_attack_fast()

    def trigger_reposition_escape(self) -> None:
        """
        Claude Escape / Reposition Combo:
        1. Skill 2 (Swap): Teleports Claude back to Dexter's shadow location
        2. Basic Attack: Instant poke on nearest enemy target
        """
        actions = [
            SkillAction(
                name="Skill 2 (Swap to Shadow)",
                coords=ClaudeConfig.COORD_SKILL_2,
                delay_after_ms=30.0,
                hold_duration_ms=50.0
            ),
            SkillAction(
                name="Basic Attack",
                coords=ClaudeConfig.COORD_BASIC_ATTACK,
                delay_after_ms=0.0,
                hold_duration_ms=45.0
            ),
        ]
        self.execute_sequence_async("ESCAPE / REPOSITION (S2 Swap -> Basic Attack)", actions)


# ==============================================================================
# KEYBOARD LISTENERS (PYNPUT + MSVCRT CONSOLE FALLBACK)
# ==============================================================================
def run_pynput_listener(controller: ClaudeMacroController):
    """Listens for global keyboard hotkeys using pynput."""
    from pynput import keyboard

    print("\n[*] Global hotkey listener active (pynput).")
    print("--------------------------------------------------")
    print("  [F] -> Engage & Ultimate (S2 -> Flicker -> Ult -> S1)")
    print("  [R] -> Escape / Reposition (S2 Swap -> Basic Attack)")
    print("  [Q] -> Safe Exit")
    print("--------------------------------------------------")

    def on_press(key):
        try:
            if hasattr(key, 'char') and key.char:
                ch = key.char.lower()
                if ch == 'f':
                    controller.trigger_engage_ultimate()
                elif ch == 'r':
                    controller.trigger_reposition_escape()
                elif ch == 'q':
                    print("\n[*] 'Q' pressed. Exiting Claude Macro cleanly.")
                    return False
        except Exception as e:
            print(f"[!] Hotkey error: {e}")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def run_msvcrt_fallback(controller: ClaudeMacroController):
    """Native Windows console keyboard fallback if pynput is not installed."""
    import msvcrt

    print("\n[*] Console keyboard listener active (msvcrt). Keep this window focused.")
    print("--------------------------------------------------")
    print("  [F] -> Engage & Ultimate (S2 -> Flicker -> Ult -> S1)")
    print("  [R] -> Escape / Reposition (S2 Swap -> Basic Attack)")
    print("  [Q] -> Safe Exit")
    print("--------------------------------------------------")

    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if ch == 'f':
                controller.trigger_engage_ultimate()
            elif ch == 'r':
                controller.trigger_reposition_escape()
            elif ch == 'q':
                print("\n[*] 'Q' pressed. Exiting Claude Macro cleanly.")
                break
        time.sleep(0.01)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    print("==================================================")
    print("   MLBB Claude (Клод) - High-Speed Combo Macro    ")
    print("==================================================")

    # Display active orientation transformation status
    print(f"[*] Orientation Mode: SWAP_XY = {ClaudeConfig.SWAP_XY}")
    sample_raw = ClaudeConfig.COORD_SKILL_1
    sample_trans = ClaudeMacroController.transform_coords(sample_raw[0], sample_raw[1])
    print(f"    Sample Skill 1: Raw {sample_raw} -> ADB Target {sample_trans}")

    controller = ClaudeMacroController(serial=ClaudeConfig.DEVICE_SERIAL)

    try:
        import pynput
        run_pynput_listener(controller)
    except ImportError:
        print("\n[!] 'pynput' not installed. Falling back to native console listener.")
        print("[!] Tip: Run 'pip install pynput' to enable global hotkeys across all windows.")
        run_msvcrt_fallback(controller)


if __name__ == "__main__":
    main()
