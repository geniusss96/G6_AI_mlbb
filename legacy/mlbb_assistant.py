"""
Mobile Legends: Bang Bang (MLBB) - Hybrid Trigger-Bot & Macro Assistant
------------------------------------------------------------------------
Engineered with Python, OpenCV, and Android Debug Bridge (ADB).

Features:
  - In-memory screencap capture (no disk I/O) via `adb exec-out screencap -p`
  - High-performance OpenCV ROI slicing & HSV color/threshold analysis
  - Asynchronous (non-blocking) input dispatch via thread worker to minimize latency
  - Chained multi-tap / combo sequences via single shell command execution
  - Safety cooldown guards and graceful keyboard shutdown ('q' or Ctrl+C)
  - Real-time debug visualization HUD with ROI bounding box
"""

import sys
import time
import subprocess
import threading
from typing import Optional, Tuple
import cv2
import numpy as np


# ==============================================================================
# CONFIGURATION & CALIBRATION (Adjust to your device resolution & settings)
# ==============================================================================
class Config:
    # --------------------------------------------------------------------------
    # Device & Screen Calibration
    # --------------------------------------------------------------------------
    # Target device resolution (Width x Height in Landscape orientation)
    SCREEN_WIDTH = 2400
    SCREEN_HEIGHT = 1080

    # Optional: ADB Device Serial (leave empty string if only one device is connected)
    DEVICE_SERIAL = ""

    # --------------------------------------------------------------------------
    # Trigger 1: Lord / Turtle HP Threshold Trigger (Retribution Smite)
    # --------------------------------------------------------------------------
    # Region of Interest (ROI) for the objective's health bar:
    # Format: (x_start, y_start, width, height)
    # Note: Calibrate using a screenshot from your device in landscape mode.
    HP_BAR_ROI = {
        "x": 1050,  # X offset of health bar
        "y": 180,   # Y offset of health bar
        "w": 300,   # Width of health bar
        "h": 15     # Height of health bar
    }

    # HSV Color range for the target health bar (e.g., Lord HP green/purple/red)
    # Calibrate these values according to the in-game health bar color
    # HSV ranges: H: 0-179, S: 0-255, V: 0-255
    HP_COLOR_LOWER = np.array([35, 100, 100], dtype=np.uint8)   # Example: Greenish
    HP_COLOR_UPPER = np.array([85, 255, 255], dtype=np.uint8)

    # Health threshold percentage (0.0 to 1.0)
    # If the remaining HP bar fill ratio drops below this value, trigger Retribution!
    # E.g., 0.12 = 12% remaining HP (approximate execute threshold)
    HP_TRIGGER_THRESHOLD = 0.12

    # Tap coordinates for Retribution (Battle Spell button)
    RETRIBUTION_TAP_COORDS = (1950, 850)

    # --------------------------------------------------------------------------
    # Trigger 2: Instant Combo Macro (Optional)
    # --------------------------------------------------------------------------
    # Sequence of taps (x, y, delay_after_ms)
    COMBO_SEQUENCE = [
        (2050, 720, 40),   # Skill 2 tap -> wait 40ms
        (1900, 600, 30),   # Skill 1 tap -> wait 30ms
        (2180, 880, 0)     # Basic Attack / Ultimate
    ]

    # --------------------------------------------------------------------------
    # Performance & Timing Safeguards
    # --------------------------------------------------------------------------
    TARGET_FPS = 30                        # Loop rate target (25-30 FPS)
    TRIGGER_COOLDOWN_SEC = 3.0             # Cooldown after firing to avoid spamming
    ENABLE_DEBUG_HUD = True                # Show real-time OpenCV window with ROI


# ==============================================================================
# ASYNCHRONOUS ADB CONTROLLER
# ==============================================================================
class ADBController:
    """Manages low-latency interaction with Android device via ADB."""

    def __init__(self, serial: str = ""):
        self.serial = serial.strip()
        self.base_cmd = ["adb"]
        if self.serial:
            self.base_cmd.extend(["-s", self.serial])

        self._verify_connection()

    def _verify_connection(self) -> None:
        """Checks if ADB daemon can see the target device."""
        try:
            result = subprocess.run(
                self.base_cmd + ["get-state"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            state = result.stdout.strip()
            if state != "device":
                raise ConnectionError(f"Device state is '{state}', expected 'device'.")
            print(f"[+] ADB Connection verified. Device status: {state}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"[!] ADB Error: Could not connect to device. Ensure ADB is in PATH and USB Debugging is ON.")
            print(f"[!] Detail: {e}")
            sys.exit(1)

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Streams a screenshot directly from device memory using `adb exec-out screencap -p`.
        Decodes in-memory via cv2.imdecode without touching disk.
        """
        try:
            # exec-out streams binary stdout without newline translation (prevents CRLF corruption on Windows)
            process = subprocess.run(
                self.base_cmd + ["exec-out", "screencap", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True
            )
            raw_bytes = process.stdout
            if not raw_bytes:
                return None

            # Decode raw PNG byte array to BGR OpenCV image
            frame = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame
        except subprocess.SubprocessError as e:
            print(f"[!] Frame capture failed: {e}")
            return None

    def fire_tap_async(self, x: int, y: int) -> None:
        """Dispatches an ADB tap command on a background thread to prevent loop stutter."""
        def _worker():
            cmd = self.base_cmd + ["shell", "input", "tap", str(x), str(y)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        threading.Thread(target=_worker, daemon=True).start()

    def fire_combo_async(self, sequence: list) -> None:
        """
        Dispatches a chained combo sequence in a single subshell session.
        Chaining commands with '&&' and 'usleep' dramatically reduces ADB roundtrip latency.
        """
        def _worker():
            # Build shell script one-liner: input tap x y && usleep microseconds && ...
            commands = []
            for item in sequence:
                x, y, delay_ms = item
                commands.append(f"input tap {x} {y}")
                if delay_ms > 0:
                    commands.append(f"usleep {delay_ms * 1000}")

            chained_cmd = " && ".join(commands)
            cmd = self.base_cmd + ["shell", chained_cmd]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        threading.Thread(target=_worker, daemon=True).start()


# ==============================================================================
# VISION & TRIGGER LOGIC
# ==============================================================================
class TriggerBot:
    """Evaluates frame regions using computer vision to determine trigger events."""

    def __init__(self, adb: ADBController, config: Config):
        self.adb = adb
        self.cfg = config
        self.last_trigger_time = 0.0

    def is_cooling_down(self) -> bool:
        """Returns True if the cooldown period is active."""
        return (time.time() - self.last_trigger_time) < self.cfg.TRIGGER_COOLDOWN_SEC

    def evaluate_hp_bar(self, frame: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Analyzes the target objective's health bar ROI.
        Calculates the percentage of remaining health using HSV color masking.

        Returns:
            should_trigger (bool): True if health is below threshold
            fill_ratio (float): Current calculated health bar fill ratio (0.0 to 1.0)
            roi_bgr (np.ndarray): Cropped ROI image for debug visualization
        """
        roi_cfg = self.cfg.HP_BAR_ROI
        h, w, _ = frame.shape

        # Bound-check coordinates
        y1 = max(0, roi_cfg["y"])
        y2 = min(h, roi_cfg["y"] + roi_cfg["h"])
        x1 = max(0, roi_cfg["x"])
        x2 = min(w, roi_cfg["x"] + roi_cfg["w"])

        roi_bgr = frame[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return False, 0.0, roi_bgr

        # Convert ROI to HSV for robust color thresholding under varying lighting
        hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Generate binary mask matching HP bar color
        mask = cv2.inRange(hsv_roi, self.cfg.HP_COLOR_LOWER, self.cfg.HP_COLOR_UPPER)

        total_pixels = mask.size
        active_pixels = cv2.countNonZero(mask)

        # Ratio of active health pixels to total ROI area
        fill_ratio = active_pixels / float(total_pixels) if total_pixels > 0 else 0.0

        # Trigger condition:
        # 1. Bar was detected (ratio > 0.02 to prevent triggering on empty air / despawned boss)
        # 2. Bar is below the execution threshold (e.g. <= 12% HP)
        should_trigger = (0.02 < fill_ratio <= self.cfg.HP_TRIGGER_THRESHOLD)

        return should_trigger, fill_ratio, roi_bgr

    def check_template_match(self, frame: np.ndarray, template: np.ndarray,
                             threshold: float = 0.85) -> bool:
        """
        Alternative trigger: match a pre-cropped template (e.g., Spell Ready indicator).
        """
        if template is None or frame is None:
            return False

        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val >= threshold


# ==============================================================================
# MAIN CONTROL LOOP
# ==============================================================================
def main():
    print("==================================================")
    print("   MLBB CV Hybrid Trigger-Bot / Macro Assistant   ")
    print("==================================================")
    print("[*] Initializing ADB connection...")

    adb = ADBController(serial=Config.DEVICE_SERIAL)
    bot = TriggerBot(adb, Config)

    frame_time_target = 1.0 / Config.TARGET_FPS
    print(f"[*] Target frame rate: {Config.TARGET_FPS} FPS (Tick: {frame_time_target*1000:.1f}ms)")
    print("[*] Controls: Press 'q' on the HUD window or Ctrl+C in terminal to stop.")

    try:
        while True:
            t_start = time.perf_counter()

            # 1. Capture screen frame directly into RAM
            frame = adb.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 2. Evaluate Trigger Conditions
            triggered, hp_ratio, roi_bgr = bot.evaluate_hp_bar(frame)

            # 3. Action Execution with Cooldown Protection
            if triggered:
                if not bot.is_cooling_down():
                    print(f"\n[!!!] TRIGGER DETECTED! Objective HP Ratio: {hp_ratio*100:.1f}%")
                    print(f"[>>>] Executing Retribution tap at {Config.RETRIBUTION_TAP_COORDS}...")

                    # Fire instant tap
                    tap_x, tap_y = Config.RETRIBUTION_TAP_COORDS
                    adb.fire_tap_async(tap_x, tap_y)

                    # Update cooldown timestamp
                    bot.last_trigger_time = time.time()
                else:
                    remaining = Config.TRIGGER_COOLDOWN_SEC - (time.time() - bot.last_trigger_time)

            # 4. Debug HUD Visualization
            if Config.ENABLE_DEBUG_HUD:
                # Draw ROI bounding box on main frame
                rx = Config.HP_BAR_ROI["x"]
                ry = Config.HP_BAR_ROI["y"]
                rw = Config.HP_BAR_ROI["w"]
                rh = Config.HP_BAR_ROI["h"]

                box_color = (0, 0, 255) if triggered else (0, 255, 0)
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)

                # Overlay status text
                cooldown_status = "COOLING DOWN" if bot.is_cooling_down() else "READY"
                status_text = f"HP Fill: {hp_ratio*100:.1f}% | Status: {cooldown_status}"
                cv2.putText(frame, status_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                # Downscale frame for preview window so it fits host monitor nicely
                preview_w = 960
                preview_h = int(frame.shape[0] * (preview_w / frame.shape[1]))
                preview_frame = cv2.resize(frame, (preview_w, preview_h))

                cv2.imshow("MLBB Assistant - Debug View (Press 'q' to Quit)", preview_frame)

                # Break on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[*] 'q' key pressed. Shutting down cleanly.")
                    break

            # 5. Performance Throttle (Maintain ~25-30 FPS)
            t_elapsed = time.perf_counter() - t_start
            sleep_duration = frame_time_target - t_elapsed
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\n[*] KeyboardInterrupt (Ctrl+C) received. Exiting...")
    finally:
        cv2.destroyAllWindows()
        print("[+] Assistant stopped successfully.")


if __name__ == "__main__":
    main()
