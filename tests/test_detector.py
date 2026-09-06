"""
Unit tests for vision.detector module.
Mocks Ultralytics YOLO to verify Detection properties and YoloDetector logic without GPU or model file.
"""

from unittest.mock import MagicMock, patch
import math
import numpy as np

from vision.detector import Detection, YoloDetector
from config.config import WEIGHTS_PATH, YOLO_DEVICE, YOLO_IMGSZ

TOLERANCE = 1e-4


def test_detection_properties():
    """Verify read-only properties: center, width, height, xywh."""
    det = Detection(
        class_id=1,
        class_name="hp_enemy",
        confidence=0.85,
        x1=100.0,
        y1=200.0,
        x2=300.0,
        y2=400.0,
    )

    assert det.class_id == 1
    assert det.class_name == "hp_enemy"
    assert math.isclose(det.confidence, 0.85, rel_tol=TOLERANCE)
    assert math.isclose(det.width, 200.0, rel_tol=TOLERANCE)
    assert math.isclose(det.height, 200.0, rel_tol=TOLERANCE)

    cx, cy = det.center
    assert math.isclose(cx, 200.0, rel_tol=TOLERANCE)
    assert math.isclose(cy, 300.0, rel_tol=TOLERANCE)

    assert det.xywh == [200.0, 300.0, 200.0, 200.0]


def test_yolo_detector_init_and_config():
    """Verify YoloDetector pulls config from config.config when not provided."""
    with patch("vision.detector.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.names = {0: "hp_self", 1: "hp_enemy"}
        mock_yolo_cls.return_value = mock_instance

        detector = YoloDetector()
        assert detector.weights_path == WEIGHTS_PATH
        assert detector.device == YOLO_DEVICE
        assert detector.imgsz == YOLO_IMGSZ
        assert detector.names[1] == "hp_enemy"
        assert detector.class_map["hp_enemy"] == 1


def test_yolo_detector_empty_result():
    """Verify empty result returns empty list cleanly."""
    with patch("vision.detector.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.names = {0: "hp_self"}
        mock_instance.predict.return_value = []
        mock_yolo_cls.return_value = mock_instance

        detector = YoloDetector()
        dummy_frame = np.zeros((720, 1544, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame)
        assert dets == []


def test_yolo_detector_conversion():
    """Verify mocked YOLO prediction converts into Detection objects."""
    with patch("vision.detector.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.names = {0: "hp_self", 1: "hp_enemy"}

        # Build mock box
        mock_box1 = MagicMock()
        mock_box1.cls = [MagicMock(item=lambda: 1)]
        mock_box1.conf = [MagicMock(item=lambda: 0.92)]
        mock_box1.xyxy = [[50.0, 60.0, 150.0, 160.0]]

        mock_box2 = MagicMock()
        mock_box2.cls = [MagicMock(item=lambda: 0)]
        mock_box2.conf = [MagicMock(item=lambda: 0.88)]
        mock_box2.xyxy = [[700.0, 300.0, 800.0, 420.0]]

        mock_res = MagicMock()
        mock_res.boxes = [mock_box1, mock_box2]
        mock_instance.predict.return_value = [mock_res]
        mock_yolo_cls.return_value = mock_instance

        detector = YoloDetector()
        dummy_frame = np.zeros((720, 1544, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame)

        assert len(dets) == 2
        assert dets[0].class_id == 1
        assert dets[0].class_name == "hp_enemy"
        assert math.isclose(dets[0].confidence, 0.92, rel_tol=TOLERANCE)
        assert dets[0].width == 100.0
        assert dets[0].height == 100.0

        assert dets[1].class_id == 0
        assert dets[1].class_name == "hp_self"
        assert math.isclose(dets[1].confidence, 0.88, rel_tol=TOLERANCE)


if __name__ == "__main__":
    test_detection_properties()
    test_yolo_detector_init_and_config()
    test_yolo_detector_empty_result()
    test_yolo_detector_conversion()
    print("All vision.detector tests passed successfully!")
