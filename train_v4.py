"""
Обучение YOLOv8 модели v4 на объединенном чистом датасете MLBB_Combined_v4
========================================================================
- Датасет: 877 картинок (без мусора с модельками героев)
- Классы: hp_self, hp_enemy, minion_enemy, minion_ally, turret_enemy, turret_ally, bush, etc.
- Архитектура: YOLOv8n (Nano) - максимальная скорость для игры в реальном времени (60+ FPS)
- Аппаратное ускорение: CUDA GPU 0 (NVIDIA P106-100 6GB)
"""

import os
from pathlib import Path
from ultralytics import YOLO

def main():
    print("==================================================")
    print("      MLBB YOLOv8 v4 Training (Mega Dataset)      ")
    print("==================================================")

    dataset_dir = Path("dataset_raw/MLBB_Combined_v4")
    yaml_path = dataset_dir / "data.yaml"

    if not yaml_path.exists():
        print(f"[ERROR] Датасет не найден: {yaml_path.absolute()}")
        return

    print(f"[*] Датасет подтвержден: {yaml_path.absolute()}")
    print("[*] Инициализация YOLOv8n (Nano)...")
    model = YOLO("yolov8n.pt")

    print("\n[*] Запуск тренировки (версия v4, 50 эпох, 877 картинок)...")
    print("[*] Ускоритель: CUDA (GPU 0 - NVIDIA P106-100)\n")

    results = model.train(
        data=str(yaml_path.absolute()),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="mlbb_training",
        name="v4_run",
        exist_ok=True,
        workers=2,
        plots=True
    )

    best_weights_path = Path("mlbb_training/v4_run/weights/best.pt")
    if not best_weights_path.exists():
        best_weights_path = Path("runs/detect/mlbb_training/v4_run/weights/best.pt")

    print("\n==================================================")
    print(f"[*] Тренировка завершена! Лучшие веса: {best_weights_path}")
    print("==================================================")

if __name__ == '__main__':
    main()
