"""Train YOLO on the audited dataset without replacing production weights.

Use python -m training.dataset_tools prepare first.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

# Support both python -m training.train_v4 and python training/train_v4.py.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from training.dataset_tools import audit, DEFAULT_OUTPUT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_OUTPUT / 'data.yaml')
    parser.add_argument('--weights', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4, help='Small default for 6 GB GPUs')
    parser.add_argument('--workers', type=int, default=0, help='Windows-compatible default')
    parser.add_argument('--device', default='auto', help='auto, cpu, or GPU index such as 0')
    parser.add_argument('--name', default='v4_clean')
    parser.add_argument('--dry-run', action='store_true', help='Audit data and print settings without loading YOLO')
    args = parser.parse_args(argv)
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0:
        parser.error('epochs, imgsz and batch must be positive; workers must be nonnegative')
    try:
        report, _ = audit(args.data)
        if report['issues'] or report['duplicate_groups'] or report['path_warnings']:
            raise ValueError('Dataset is not clean/portable. Run python -m training.dataset_tools prepare first.')
        if not report['images']['train'] or not report['images']['val']:
            raise ValueError('Training requires nonempty train and val splits')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Dataset check failed: {error}\n')
    settings = dict(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, workers=args.workers, device=args.device,
                    project=str(ROOT / 'runs/detect'), name=args.name,
                    exist_ok=False, seed=0, plots=True)
    print(json.dumps(settings, indent=2))
    if args.dry_run:
        print('Dataset check passed. No model loaded and no training started.')
        return 0
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        parser.exit(1, f'Missing training dependency: {error}. Install project requirements first.\n')
    if args.device == 'auto':
        settings['device'] = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Training device: {settings['device']}")
    model = YOLO(args.weights)
    model.train(**settings)
    print(f"Training finished. Outputs: {model.trainer.save_dir}")
    print('Production weights were not replaced. Validate the new model before selecting it.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
