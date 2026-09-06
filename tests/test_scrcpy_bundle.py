"""
tests/test_scrcpy_bundle.py — Validation of the bundled Scrcpy tools & configuration.
Verifies existence, paths, and DLLs/resources of the project-local tools/scrcpy bundle.
Completely offline, zero dependency on real Android devices.
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from config.config import (
    PROJECT_ROOT,
    SCRCPY_DIR,
    SCRCPY_PATH,
    ADB_PATH,
    ADB_BINARY,
)
from control.adb import ADBTransport


class TestScrcpyBundle(unittest.TestCase):
    def test_configured_paths_resolution(self):
        """Verify project-relative resolution of tools/scrcpy paths."""
        self.assertTrue(os.path.isabs(SCRCPY_DIR))
        self.assertTrue(os.path.isabs(SCRCPY_PATH))
        self.assertTrue(os.path.isabs(ADB_PATH))
        self.assertEqual(SCRCPY_DIR, os.path.join(PROJECT_ROOT, "tools", "scrcpy"))
        self.assertEqual(SCRCPY_PATH, os.path.join(SCRCPY_DIR, "scrcpy.exe"))
        self.assertEqual(ADB_PATH, os.path.join(SCRCPY_DIR, "adb.exe"))

    def test_bundle_binaries_exist(self):
        """Verify that core scrcpy and adb executables exist in tools/scrcpy/."""
        self.assertTrue(os.path.isfile(SCRCPY_PATH), f"scrcpy.exe missing at {SCRCPY_PATH}")
        self.assertTrue(os.path.isfile(ADB_PATH), f"adb.exe missing at {ADB_PATH}")

    def test_bundle_essential_dlls_exist(self):
        """Verify required DLLs and assets exist in the bundle."""
        required_files = [
            "AdbWinApi.dll",
            "AdbWinUsbApi.dll",
            "SDL3.dll",
            "libusb-1.0.dll",
            "avcodec-62.dll",
            "avformat-62.dll",
            "avutil-60.dll",
            "swresample-6.dll",
            "scrcpy-server",
        ]
        for fname in required_files:
            fpath = os.path.join(SCRCPY_DIR, fname)
            self.assertTrue(os.path.isfile(fpath), f"Required bundle file {fname} missing at {fpath}")

    def test_adb_binary_points_to_bundled_adb(self):
        """Verify that default ADB_BINARY resolves to the bundled adb."""
        self.assertEqual(ADB_BINARY, ADB_PATH)
        self.assertTrue(os.path.isfile(ADB_BINARY))

    def test_root_legacy_adb_files_preserved(self):
        """Verify root adb files remain intact for backwards compatibility."""
        root_adb = os.path.join(PROJECT_ROOT, "adb.exe")
        root_winapi = os.path.join(PROJECT_ROOT, "AdbWinApi.dll")
        root_usbapi = os.path.join(PROJECT_ROOT, "AdbWinUsbApi.dll")
        self.assertTrue(os.path.isfile(root_adb), "Root legacy adb.exe missing")
        self.assertTrue(os.path.isfile(root_winapi), "Root legacy AdbWinApi.dll missing")
        self.assertTrue(os.path.isfile(root_usbapi), "Root legacy AdbWinUsbApi.dll missing")


if __name__ == "__main__":
    unittest.main()
