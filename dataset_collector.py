"""
Mobile Legends: YOLOv8 Training Data Collector
========================================================================
A lightweight, low-overhead background script to collect raw frames for 
training a custom YOLO object detection model. 

It dynamically tracks the Scrcpy window and uses `win32gui`'s ClientRect 
to extract ONLY the raw game rendering area, perfectly cropping out Windows 
title bars, window shadows, and borders to prevent AI confusion.
"""

import os
import time
import cv2
import numpy as np
import mss
import win32gui


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Search keyword for the Scrcpy window title. 
# Defaults to "SM-S908N" for the user's specific device model in Scrcpy.
WINDOW_TITLE_KEYWORD = "SM-S908N"

# Capture interval in seconds
INTERVAL = 1.5

# Directory to store the raw images
SAVE_DIR = "dataset_raw"

# JPEG Quality (0-100). 90 is a good balance between file size and quality for AI training.
JPEG_QUALITY = 90


def get_window_client_area(keyword):
    """
    Finds a window by keyword and returns its exact Client Area (rendering region).
    This securely ignores Windows 10/11 invisible borders, shadows, and the top Title Bar.
    """
    hwnds = []
    
    # Callback to search through all open windows
    def enum_windows_callback(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).lower()
            if keyword.lower() in title:
                hwnds.append(hwnd)
        return True
        
    win32gui.EnumWindows(enum_windows_callback, None)
    
    if not hwnds:
        return None
        
    # Take the first matching window
    target_hwnd = hwnds[0]
    
    # GetClientRect returns the logical rendering area (0, 0, width, height)
    left, top, right, bottom = win32gui.GetClientRect(target_hwnd)
    
    if right == 0 or bottom == 0:
        return None  # Window is likely minimized
        
    # Convert logical client coordinates to absolute screen coordinates
    screen_left, screen_top = win32gui.ClientToScreen(target_hwnd, (left, top))
    screen_right, screen_bottom = win32gui.ClientToScreen(target_hwnd, (right, bottom))
    
    return {
        "top": screen_top,
        "left": screen_left,
        "width": screen_right - screen_left,
        "height": screen_bottom - screen_top
    }


def main():
    print("==================================================")
    print("      MLBB Training Data Collector (YOLOv8)       ")
    print("==================================================")
    
    # Create the dataset directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"[*] Storage Directory : ./{SAVE_DIR}/")
    print(f"[*] Capture Interval  : {INTERVAL} seconds")
    print(f"[*] Target Window     : '{WINDOW_TITLE_KEYWORD}'")
    print("[*] Press Ctrl+C in this terminal to STOP collecting.\n")

    counter = 1
    
    try:
        with mss.mss() as sct:
            while True:
                loop_start_time = time.time()
                
                # 1. Dynamically get the exact active window region on every frame.
                # This allows you to move the Scrcpy window around while the script runs!
                monitor = get_window_client_area(WINDOW_TITLE_KEYWORD)
                
                if monitor is None:
                    print(f"\r[!] Window '{WINDOW_TITLE_KEYWORD}' not found or minimized. Waiting...", end="", flush=True)
                    time.sleep(1.0)
                    continue
                
                # 2. Capture the exact window client area
                sct_img = sct.grab(monitor)
                
                # 3. Convert to numpy array and strip alpha channel to BGR
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # 4. Generate unique filename based on Unix timestamp and counter
                timestamp = int(time.time())
                filename = f"mlbb_{timestamp}_{counter}.jpg"
                filepath = os.path.join(SAVE_DIR, filename)
                
                # 5. Save the image to disk
                cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                
                # Print progress (overwrite the same line to keep terminal clean)
                print(f"\r[+] Saved: {filename} (Total: {counter}) | Res: {monitor['width']}x{monitor['height']}", end="", flush=True)
                
                counter += 1
                
                # 6. Calculate precise sleep time to maintain exactly the requested INTERVAL
                elapsed = time.time() - loop_start_time
                sleep_time = INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        print(f"\n\n[INFO] Data collection stopped by user.")
        print(f"[INFO] Successfully saved {counter - 1} images to ./{SAVE_DIR}/")
        print("[INFO] Ready for labeling!")


if __name__ == "__main__":
    main()
