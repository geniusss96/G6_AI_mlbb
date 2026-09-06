"""
Helper Calibration Tool: Grab a frame and inspect coordinates / ROI
------------------------------------------------------------------
Usage:
  Run this script while connected to your phone in MLBB (e.g. in Practice / Training mode).
  - It saves a screenshot named 'reference_screen.png'
  - It opens an interactive window: click anywhere to see the exact (x, y) coordinates and BGR/HSV colors.
  - Drag a box to inspect ROI coordinates for Config.HP_BAR_ROI.
"""

import subprocess
import cv2
import numpy as np


def grab_single_frame():
    cmd = ["adb", "exec-out", "screencap", "-p"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
    frame = cv2.imdecode(np.frombuffer(proc.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame


def mouse_callback(event, x, y, flags, param):
    frame = param["frame"]
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = frame[y, x]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"\n[+] Clicked Coordinate: X={x}, Y={y}")
        print(f"    BGR: {bgr.tolist()} | HSV: {hsv.tolist()} (H: {hsv[0]}, S: {hsv[1]}, V: {hsv[2]})")


def main():
    print("[*] Grabbing screenshot via ADB...")
    frame = grab_single_frame()
    if frame is None:
        print("[!] Failed to capture frame.")
        return

    cv2.imwrite("reference_screen.png", frame)
    print(f"[+] Saved 'reference_screen.png'. Dimensions: {frame.shape[1]}x{frame.shape[0]}")
    print("[*] Click on elements (buttons, HP bars) to view coordinates & HSV color values.")
    print("[*] Press 's' to select an ROI with drag-and-drop, or 'q' to exit.")

    window_name = "Calibration - Click for Coordinates / 's' for ROI / 'q' to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, {"frame": frame})

    while True:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("\n[*] Drag a box around the target region (e.g., Lord HP Bar), then press ENTER or SPACE:")
            r = cv2.selectROI(window_name, frame)
            if r[2] > 0 and r[3] > 0:
                print(f"[+] Selected ROI -> x: {r[0]}, y: {r[1]}, w: {r[2]}, h: {r[3]}")
                print(f"    Paste into Config.HP_BAR_ROI: {{'x': {r[0]}, 'y': {r[1]}, 'w': {r[2]}, 'h': {r[3]}}}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
