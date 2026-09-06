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
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
    ):
        self.weights_path = weights_path or WEIGHTS_PATH
        self.device = device or YOLO_DEVICE
        self.imgsz = imgsz or YOLO_IMGSZ

        # Load Ultralytics model
        self.model = YOLO(self.weights_path)
        self.class_map: Dict[str, int] = {v: k for k, v in self.model.names.items()}
        self.names: Dict[int, str] = dict(self.model.names)

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
            half: Use FP16 half-precision inference.

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
