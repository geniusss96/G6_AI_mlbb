"""
Mobile Legends: Bang Bang - Claude (Клод) CV Trigger Bot
========================================================================
A real-time AI automated trigger bot that interfaces with the ClaudeMacroController.
It captures the screen using `mss` and uses a Deep Learning Object Detection 
model (YOLOv8) to semantically identify Enemy Heroes vs Creeps/UI.

Architecture:
  - Screen Capture: mss combined with pygetwindow for dynamic bounds.
  - Neural Network: ultralytics YOLOv8 for real-time object detection.
  - Semantic Filtering: Ignores all detected classes except `enemy_hero`.
  - Action & Cooldown: Asynchronously triggers `ClaudeMacroController`.
"""

import time
import os
import cv2
import numpy as np
import mss
import pygetwindow as gw
from ultralytics import YOLO
from claude_macro import ClaudeMacroController, ClaudeConfig


# ==============================================================================
# AI TARGETING & CONFIGURATION
# ==============================================================================
class CVBotConfig:
    # --------------------------------------------------------------------------
    # Window Tracking
    # --------------------------------------------------------------------------
    WINDOW_TITLE = "SM-S908N"  # Matches Scrcpy window title. Leave "" for full monitor capture
    MONITOR = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    # --------------------------------------------------------------------------
    # Deep Learning Model (YOLOv8)
    # --------------------------------------------------------------------------
    # You MUST provide a trained weights file (.pt). 
    # Without this "brain", the bot cannot distinguish heroes from creeps.
    MODEL_PATH = "mlbb_custom_v8.pt"
    
    # Neural Network Confidence & Classes
    # Class 0: enemy_hero | Class 1: creep | Class 2: ui_button
    ENEMY_HERO_CLASS_ID = 0
    CONF_THRESHOLD = 0.75  # 75% certainty required to trigger

    # --------------------------------------------------------------------------
    # Engage Range (Kill Zone)
    # --------------------------------------------------------------------------
    # Only fire if the AI detects the hero inside this central bounding box.
    KILL_ZONE_WIDTH = 600
    KILL_ZONE_HEIGHT = 600

    # --------------------------------------------------------------------------
    # Action Cooldown
    # --------------------------------------------------------------------------
    COOLDOWN_SECONDS = 25.0


def check_intersection(rect1, rect2) -> bool:
    """Checks if two rectangles intersect (x1, y1, x2, y2)."""
    r1_x1, r1_y1, r1_x2, r1_y2 = rect1
    r2_x1, r2_y1, r2_x2, r2_y2 = rect2
    overlap_x = max(0, min(r1_x2, r2_x2) - max(r1_x1, r2_x1))
    overlap_y = max(0, min(r1_y2, r2_y2) - max(r1_y1, r2_y1))
    return overlap_x > 0 and overlap_y > 0


def main():
    print("==================================================")
    print("   MLBB Claude AI Bot (YOLOv8 Neural Network)     ")
    print("==================================================")

    # 1. Setup Window Tracking
    if CVBotConfig.WINDOW_TITLE:
        try:
            windows = gw.getWindowsWithTitle(CVBotConfig.WINDOW_TITLE)
            if windows:
                win = windows[0]
                CVBotConfig.MONITOR = {
                    "top": win.top, "left": win.left, 
                    "width": win.width, "height": win.height
                }
                print(f"[*] Found Window '{CVBotConfig.WINDOW_TITLE}'")
        except Exception as e:
            print(f"[!] Error tracking window: {e}. Defaulting to full screen.")

    actual_frame_width = CVBotConfig.MONITOR["width"]
    actual_frame_height = CVBotConfig.MONITOR["height"]

    # 2. Load Neural Network
    print(f"[*] Loading YOLOv8 Brain: {CVBotConfig.MODEL_PATH}...")
    if not os.path.exists(CVBotConfig.MODEL_PATH):
        print(f"[ERROR] Weights file '{CVBotConfig.MODEL_PATH}' not found!")
        print("[ERROR] Please provide the trained .pt file to run the AI.")
        return

    model = YOLO(CVBotConfig.MODEL_PATH)
    # Optimize for TensorRT/Half-precision if available (optional for speed)
    # model.to('cuda') 

    # 3. Initialize ADB Macro Controller
    print("[*] Initializing Claude ADB Macro Controller...")
    controller = ClaudeMacroController(serial=ClaudeConfig.DEVICE_SERIAL)
    
    last_execution_time = 0.0

    # Calculate Kill Zone based on current window size
    center_x = actual_frame_width // 2
    center_y = actual_frame_height // 2
    kz_x1 = center_x - (CVBotConfig.KILL_ZONE_WIDTH // 2)
    kz_y1 = center_y - (CVBotConfig.KILL_ZONE_HEIGHT // 2)
    kz_x2 = center_x + (CVBotConfig.KILL_ZONE_WIDTH // 2)
    kz_y2 = center_y + (CVBotConfig.KILL_ZONE_HEIGHT // 2)
    kill_zone_rect = (kz_x1, kz_y1, kz_x2, kz_y2)

    print(f"[*] Action Cooldown: {CVBotConfig.COOLDOWN_SECONDS} seconds.")
    print("[*] Starting AI vision loop... Press 'q' to exit.")

    # Initialize mss screen capture
    with mss.mss() as sct:
        while True:
            start_time = time.time()
            
            # Capture the screen
            sct_img = sct.grab(CVBotConfig.MONITOR)
            frame = np.array(sct_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = np.ascontiguousarray(frame)

            cooldown_active = (time.time() - last_execution_time) < CVBotConfig.COOLDOWN_SECONDS
            target_acquired = False
            best_conf = 0.0

            # 4. Neural Network Inference
            # verbose=False prevents YOLO from printing stats to console every frame
            results = model.predict(frame, verbose=False, imgsz=640)
            
            # Draw Kill Zone
            kz_color = (0, 255, 255) if cooldown_active else (255, 0, 0)
            cv2.rectangle(frame, (kz_x1, kz_y1), (kz_x2, kz_y2), kz_color, 2)
            cv2.putText(frame, "ENGAGE ZONE", (kz_x1, kz_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, kz_color, 2)

            # 5. Semantic Filtering (Parse YOLO Bounding Boxes)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # FILTER: Ignore creeps (1), UI (2), etc. ONLY process enemy_hero (0)
                    if cls_id != CVBotConfig.ENEMY_HERO_CLASS_ID:
                        continue

                    # FILTER: Confidence threshold
                    if conf < CVBotConfig.CONF_THRESHOLD:
                        continue
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    enemy_rect = (x1, y1, x2, y2)

                    # Draw generic detection box (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"HERO {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if the valid hero is inside our Engage/Kill Zone
                    if check_intersection(enemy_rect, kill_zone_rect):
                        target_acquired = True
                        best_conf = max(best_conf, conf)
                        # Highlight targeted enemy in MAGENTA
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # 6. Strict Dual Condition Trigger Logic
            is_cooldown_ready = not cooldown_active
            is_target_present = target_acquired

            # Execute ONLY when BOTH conditions are true simultaneously.
            if is_cooldown_ready and is_target_present:
                print(f"\n[!!!] ENEMY HERO LOCKED! (AI Confidence: {best_conf:.1%})")
                print(f"[>>>] Triggering Claude Engage Ultimate Combo...")
                controller.trigger_engage_ultimate()
                last_execution_time = time.time()
            
            # HUD Info
            fps = 1.0 / (time.time() - start_time)
            hud_status = "READY"
            if cooldown_active:
                remaining = CVBotConfig.COOLDOWN_SECONDS - (time.time() - last_execution_time)
                hud_status = f"COOLDOWN ({remaining:.1f}s)"
                
            cv2.putText(frame, f"AI FPS: {fps:.1f} | Status: {hud_status}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Display Feed
            debug_frame = cv2.resize(frame, (CVBotConfig.DISPLAY_WIDTH, CVBotConfig.DISPLAY_HEIGHT))
            cv2.imshow(window_name, debug_frame)

            # Clean Exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[*] 'q' pressed. Shutting down AI bot.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
