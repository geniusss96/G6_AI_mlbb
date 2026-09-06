"""
Mobile Legends: Video to YOLO Dataset Extractor
========================================================================
A high-performance script to automatically extract training frames from 
gameplay videos (.mp4, .mkv, .avi) for YOLOv8 model training.

Features:
- Interval-based extraction (e.g., every 1.5 seconds)
- Auto-Letterbox resize to 640x640 (standard YOLO resolution)
- Black/empty frame filtering
- Fast frame jumping via cv2.CAP_PROP_POS_FRAMES
"""

import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_DIR = "input_videos"        # Folder containing your gameplay recordings
OUTPUT_DIR = "dataset_raw"        # Folder where frames will be saved
INTERVAL_SECONDS = 1.5            # Take a screenshot every X seconds

# YOLO Optimization
RESIZE_TO_640 = True              # If True, resizes to 640x640 with padding (Letterbox)
YOLO_SIZE = 640
PAD_COLOR = (114, 114, 114)       # Standard YOLO padding color (Gray)
JPEG_QUALITY = 95                 # High quality to avoid motion blur artifacts (0-100)

# Filters
BLACK_SCREEN_THRESH = 10.0        # If the average pixel brightness is below this, discard (loading screens)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def letterbox_image(image, target_size, pad_color):
    """
    Resizes an image to a target square size while maintaining aspect ratio.
    Pads the remaining space with a solid color.
    """
    shape = image.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(target_size / shape[0], target_size / shape[1])
    
    # Compute new unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Calculate padding
    dw, dh = target_size - new_unpad[0], target_size - new_unpad[1]
    
    # Divide padding into two sides (top/bottom, left/right)
    dw /= 2
    dh /= 2

    # Resize if necessary
    if shape[::-1] != new_unpad:
        # INTER_LINEAR is faster but still good quality for downsampling
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Add border
    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded_img

def is_valid_frame(image, threshold=BLACK_SCREEN_THRESH):
    """
    Checks if the frame is completely black or an empty loading screen.
    """
    # Convert to grayscale to check overall brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    
    # If the average brightness is too low, it's a black screen
    if mean_val < threshold:
        return False
    return True

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def process_video(video_path: Path, out_dir: Path):
    """Processes a single video file and extracts frames."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"[!] Error opening video file: {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        print(f"[!] Invalid video metadata: {video_path.name}")
        cap.release()
        return 0

    # Calculate how many frames to skip based on the requested interval
    frame_interval = int(fps * INTERVAL_SECONDS)
    if frame_interval <= 0:
        frame_interval = 1

    vid_name = video_path.stem.replace(" ", "_")
    saved_count = 0
    
    # Process frames using tqdm for a progress bar
    # We step by frame_interval
    for frame_idx in tqdm(range(0, total_frames, frame_interval), desc=f"Processing {video_path.name}", unit="frame"):
        
        # Fast-forward to the exact frame we need
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            break
            
        # 1. Filter out completely black screens
        if not is_valid_frame(frame):
            continue
            
        # 2. Resize and pad for YOLO (if enabled)
        if RESIZE_TO_640:
            frame = letterbox_image(frame, target_size=YOLO_SIZE, pad_color=PAD_COLOR)
            
        # 3. Create unique filename: vid_name_timestamp_frame_index.jpg
        # Using the video timestamp in seconds for context
        vid_timestamp_sec = int(frame_idx / fps)
        out_filename = f"{vid_name}_T{vid_timestamp_sec}s_F{frame_idx}.jpg"
        out_filepath = out_dir / out_filename
        
        # 4. Save to disk
        cv2.imwrite(str(out_filepath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        saved_count += 1

    cap.release()
    return saved_count


def main():
    print("==================================================")
    print("     MLBB Video-to-Dataset YOLOv8 Extractor       ")
    print("==================================================")
    
    input_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    
    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan for valid video files
    video_extensions = {".mp4", ".mkv", ".avi", ".mov"}
    video_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"[!] No video files found in ./{INPUT_DIR}/")
        print("[*] Please place your MLBB gameplay recordings in that folder and run again.")
        return

    print(f"[*] Found {len(video_files)} videos in ./{INPUT_DIR}/")
    print(f"[*] Extraction Interval : Every {INTERVAL_SECONDS}s")
    print(f"[*] YOLO Letterboxing   : {'ON (640x640)' if RESIZE_TO_640 else 'OFF'}")
    print(f"[*] Output Directory    : ./{OUTPUT_DIR}/\n")
    
    total_saved = 0
    start_time = time.time()
    
    for video in video_files:
        saved = process_video(video, out_dir)
        total_saved += saved
        
    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f"[SUCCESS] Extraction Complete in {elapsed:.1f} seconds!")
    print(f"[*] Total frames extracted and saved: {total_saved}")
    print(f"[*] Ready for YOLO labeling in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
