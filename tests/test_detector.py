"""
Unit tests for vision.detector module.
Mocks Ultralytics YOLO to verify Detection properties and YoloDetector logic without GPU or model file.
Stage 26: added GPU device configuration tests (no physical GPU required).
"""

import unittest
import warnings
from unittest.mock import MagicMock, patch
import math
import numpy as np

from vision.detector import Detection, YoloDetector
from config.config import WEIGHTS_PATH, YOLO_DEVICE, YOLO_IMGSZ, YOLO_CUDA_STRICT

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
    with patch("vision.detector.YOLO") as mock_yolo_cls, \
         patch("torch.cuda.is_available", return_value=True):
        mock_instance = MagicMock()
        mock_instance.names = {0: "hp_self", 1: "hp_enemy"}
        mock_yolo_cls.return_value = mock_instance

        detector = YoloDetector(device="cuda:0", cuda_strict=False)
        assert detector.weights_path == WEIGHTS_PATH
        assert detector.device == "cuda:0"
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

        detector = YoloDetector(device="cpu", cuda_strict=False)
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

        detector = YoloDetector(device="cpu", cuda_strict=False)
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


# ---------------------------------------------------------------------------
# Stage 26 — GPU device configuration tests (no physical GPU required)
# ---------------------------------------------------------------------------

class TestCudaDeviceConfig(unittest.TestCase):
    """Verify GPU device configuration and unavailability handling."""

    def test_config_yolo_device_is_cuda_or_cpu(self):
        """YOLO_DEVICE must be an explicit device string (not the ambiguous '0')."""
        import os
        if "MLBB_YOLO_DEVICE" not in os.environ:
            self.assertTrue(
                YOLO_DEVICE.startswith("cuda") or YOLO_DEVICE == "cpu",
                f"YOLO_DEVICE should be 'cuda:0' or 'cpu', got {YOLO_DEVICE!r}. "
                f"The ambiguous '0' string is not acceptable."
            )

    def test_config_yolo_cuda_strict_is_bool(self):
        """YOLO_CUDA_STRICT must be a boolean."""
        self.assertIsInstance(YOLO_CUDA_STRICT, bool)

    def test_cuda_strict_raises_when_cuda_unavailable(self):
        """
        YoloDetector must raise RuntimeError when CUDA is explicitly configured
        but torch.cuda.is_available() returns False and cuda_strict=True.
        No silent CPU fallback in strict mode.
        """
        with patch("vision.detector.YOLO") as mock_yolo_cls, \
             patch("torch.cuda.is_available", return_value=False):
            mock_instance = MagicMock()
            mock_instance.names = {0: "hp_self"}
            mock_yolo_cls.return_value = mock_instance

            with self.assertRaises(RuntimeError) as ctx:
                YoloDetector(device="cuda:0", cuda_strict=True)

            self.assertIn("CUDA device requested", str(ctx.exception))
            self.assertIn("cuda:0", str(ctx.exception))

    def test_non_strict_warns_and_falls_back_when_cuda_unavailable(self):
        """
        When cuda_strict=False and CUDA is unavailable, YoloDetector must
        emit a RuntimeWarning and fall back to 'cpu' — no silent failure.
        """
        with patch("vision.detector.YOLO") as mock_yolo_cls, \
             patch("torch.cuda.is_available", return_value=False):
            mock_instance = MagicMock()
            mock_instance.names = {0: "hp_self"}
            mock_yolo_cls.return_value = mock_instance

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                detector = YoloDetector(device="cuda:0", cuda_strict=False)

            self.assertEqual(detector.device, "cpu", "Should fall back to CPU")
            warning_msgs = [str(warning.message) for warning in w]
            self.assertTrue(
                any("CUDA" in msg for msg in warning_msgs),
                f"Expected CUDA fallback warning, got: {warning_msgs}"
            )

    def test_cpu_device_skips_cuda_check(self):
        """
        When device='cpu' is explicitly set, no CUDA check should block
        initialisation regardless of GPU availability.
        """
        with patch("vision.detector.YOLO") as mock_yolo_cls, \
             patch("torch.cuda.is_available", return_value=False):
            mock_instance = MagicMock()
            mock_instance.names = {0: "hp_self"}
            mock_yolo_cls.return_value = mock_instance

            # Should NOT raise even though CUDA is unavailable
            detector = YoloDetector(device="cpu", cuda_strict=True)
            self.assertEqual(detector.device, "cpu")


if __name__ == "__main__":
    test_detection_properties()
    test_yolo_detector_init_and_config()
    test_yolo_detector_empty_result()
    test_yolo_detector_conversion()
    unittest.main(argv=[""], exit=False, verbosity=2)
    print("All vision.detector tests passed successfully!")
