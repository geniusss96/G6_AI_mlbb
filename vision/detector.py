"""
vision.detector — YOLOv8 object detector for MLBB battlefield entities.

Isolates neural network inference, confidence filtering, and provides
standardized Detection dataclass objects in reference coordinates (1544x720).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from ultralytics import YOLO

from config.config import (
    WEIGHTS_PATH,
    YOLO_DEVICE,
    YOLO_CUDA_STRICT,
    YOLO_IMGSZ,
    CONF_HERO_THRESHOLD,
    CONF_BUFF_THRESHOLD,
)


@dataclass(frozen=True)
class Detection:
    """
    Unified computer vision bounding box detection on reference resolution (1544x720).
    Pure CV descriptor without tactical state or business logic.
    """
    class_id: int
    class_name: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> Tuple[float, float]:
        """(x_center, y_center)"""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def xywh(self) -> List[float]:
        """Format [cx, cy, w, h] commonly used in MLBB CV pipeline."""
        cx, cy = self.center
        return [cx, cy, self.width, self.height]


class YoloDetector:
    """
    YOLOv8 inference wrapper for MLBB battlefield entity recognition.

    Device selection:
        - Configured via YOLO_DEVICE (default "cuda:0").
        - Overridable at construction time via the `device` argument.
        - If CUDA is requested but unavailable, raises RuntimeError when
          YOLO_CUDA_STRICT is True (production default).
          Set MLBB_YOLO_CUDA_STRICT=0 to allow CPU fallback in CI/offline envs.
        - Effective inference device is confirmed after the first .predict() call
          (Ultralytics transfers model parameters to the configured device lazily).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        cuda_strict: Optional[bool] = None,
    ):
        self.weights_path = weights_path or WEIGHTS_PATH
        self.device = device or YOLO_DEVICE
        self.imgsz = imgsz or YOLO_IMGSZ
        self._cuda_strict = cuda_strict if cuda_strict is not None else YOLO_CUDA_STRICT

        # --- CUDA availability guard ---
        # Check before model load so we fail early with a clear message.
        _is_cuda_device = (
            isinstance(self.device, str)
            and (self.device.startswith("cuda") or (self.device.isdigit() and self.device != "cpu"))
        )
        if _is_cuda_device:
            try:
                import torch
                _cuda_available = torch.cuda.is_available()
            except ImportError:
                _cuda_available = False

            if not _cuda_available:
                if self._cuda_strict:
                    raise RuntimeError(
                        f"[YoloDetector] CUDA device requested ({self.device!r}) "
                        f"but torch.cuda.is_available() is False. "
                        f"Install a CUDA-enabled PyTorch build, or set "
                        f"MLBB_YOLO_CUDA_STRICT=0 and MLBB_YOLO_DEVICE=cpu "
                        f"to allow CPU fallback."
                    )
                # Non-strict mode: fall back to CPU with a clear warning
                import warnings
                warnings.warn(
                    f"[YoloDetector] CUDA requested ({self.device!r}) but CUDA is "
                    f"unavailable. Falling back to 'cpu' (YOLO_CUDA_STRICT=False).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.device = "cpu"

        # Load Ultralytics model
        self.model = YOLO(self.weights_path)
        self.class_map: Dict[str, int] = {v: k for k, v in self.model.names.items()}
        self.names: Dict[int, str] = dict(self.model.names)

        # Startup device diagnostic — printed at construction time
        print(
            f"[YoloDetector] configured device = {self.device!r} | "
            f"weights = {self.weights_path}"
        )
        print(
            "[YoloDetector] Note: model parameters move to the configured device "
            "on the first .predict() call. Verify effective device after first inference."
        )

    def detect(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        half: bool = True,
    ) -> List[Detection]:
        """
        Runs YOLOv8 forward pass on reference frame (1544x720).

        Args:
            frame: Pre-normalized BGR image (1544x720).
            conf: Min confidence cutoff for inference. Defaults to min(CONF_HERO_THRESHOLD, CONF_BUFF_THRESHOLD).
            half: Use FP16 half-precision inference (supported on CUDA; auto-disabled on CPU).

        Returns:
            List of Detection objects.
        """
        if conf is None:
            conf = min(CONF_HERO_THRESHOLD, CONF_BUFF_THRESHOLD)

        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=conf,
            verbose=False,
            imgsz=self.imgsz,
            half=half,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
            score = float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0])
            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
            class_name = self.names.get(cls_id, f"class_{cls_id}")

            det = Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=score,
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )
            detections.append(det)

        return detections
