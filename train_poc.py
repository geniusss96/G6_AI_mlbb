import os
import random
import cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    print("==================================================")
    print("      MLBB YOLOv8 Proof of Concept Training       ")
    print("==================================================")
    
    # 1. Path Configuration
    dataset_dir = Path("dataset_raw/MLBB_New_Dataset")
    yaml_path = dataset_dir / "data.yaml"
    
    # Check if dataset configuration exists (Try-Except block as requested)
    try:
        if not yaml_path.exists():
            raise FileNotFoundError(f"Cannot find dataset configuration file at:\n{yaml_path.absolute()}")
    except Exception as e:
        print(f"\n[ERROR] Dataset validation failed:\n{e}")
        print("\nPlease ensure you have extracted your dataset correctly and the folder structure matches.")
        return

    print(f"[*] Dataset verified at: {yaml_path.absolute()}")
    print("[*] Initializing lightweight YOLOv8 Nano model (yolov8n.pt)...")
    
    # 2. Initialize Model
    # Download the foundational weights for YOLOv8-nano (fastest model, great for gaming bots)
    model = YOLO("yolov8n.pt")  
    
    print("\n[*] Starting Training Phase (Version 3 Dataset)...")
    print("[*] Epochs: 50 | Image Size: 640 | Hardware: CUDA (GPU 0)")
    print("[*] Note: If this is your first run, downloading yolov8n.pt may take a few seconds.\n")
    
    # 3. Training Block
    # Explicitly routing computations to CUDA via `device=0` for your NVIDIA P104-100 / P106-100
    results = model.train(
        data=str(yaml_path.absolute()), 
        epochs=50, 
        imgsz=640, 
        device=0,                   # Force CUDA / GPU processing
        project="mlbb_training",    # Custom folder for clean output management
        name="v3_run",              # Subfolder name for this Version 3 run
        exist_ok=True               # Overwrite if we run the script multiple times
    )
    
    # 4. Visualization & Inference Block
    print("\n==================================================")
    print("           Training Complete! Testing Model       ")
    print("==================================================")
    
    # Dynamically locate the saved weights from the trainer instance or fallback paths
    best_weights_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights_path.exists():
        fallback = Path("runs/detect/mlbb_training/v3_run/weights/best.pt")
        if fallback.exists():
            best_weights_path = fallback
        else:
            print(f"[ERROR] Could not find the trained weights at {best_weights_path}")
            return
        
    print(f"[*] Loading best weights from: {best_weights_path}")
    best_model = YOLO(str(best_weights_path))
    
    # Locate test images directory
    test_images_dir = dataset_dir / "test" / "images"
    
    # Fallback in case the user's dataset only has train/valid splits
    if not test_images_dir.exists():
        print(f"[!] 'test' folder not found. Falling back to 'valid/images'...")
        test_images_dir = dataset_dir / "valid" / "images"
        
    if not test_images_dir.exists():
        print("[ERROR] Could not find any test or validation images for visual prediction.")
        return
        
    # Gather all image files
    valid_extensions = ['.jpg', '.jpeg', '.png']
    test_images = [f for f in test_images_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    if not test_images:
        print(f"[ERROR] No images found inside {test_images_dir}.")
        return
        
    # Pick a random image for the Proof of Concept test
    random_image_path = random.choice(test_images)
    print(f"[*] Selected random test image: {random_image_path.name}")
    
    # Run prediction (inference) on the selected image
    print("[*] Running inference...")
    predict_results = best_model.predict(source=str(random_image_path), imgsz=640, conf=0.25)
    
    # 5. Output rendering
    # predict_results[0].plot() draws the bounding boxes and labels automatically onto the image array
    res_plotted = predict_results[0].plot()
    
    # Save the plotted image locally
    output_test_file = "poc_test_result.jpg"
    cv2.imwrite(output_test_file, res_plotted)
    print(f"[+] Prediction saved to ./{output_test_file}")
    
    # Display the result to the user
    print("[*] Opening visualization window...")
    print("[!] Press ANY KEY on the image window to close it and exit.")
    
    cv2.imshow("YOLOv8 MLBB - Proof of Concept Result", res_plotted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Wrapping in __main__ is absolutely critical on Windows for multiprocessing compatibility
if __name__ == '__main__':
    main()
