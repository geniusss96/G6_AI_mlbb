"""
Unit tests for control.adb (ADBTransport).
Tests execute deterministically without physical Android devices using mocks.
Compatible with pytest, unittest discover, and direct script execution.
"""

import os
import sys
import subprocess
import unittest
from unittest.mock import MagicMock, patch

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from config.config import ADB_BINARY, DEVICE_SERIAL
from control.adb import ADBTransport


class FakeStdin:
    def __init__(self):
        self.written = []
        self.flushed = 0

    def write(self, data: str):
        self.written.append(data)

    def flush(self):
        self.flushed += 1


class FakeProcess:
    def __init__(self):
        self.stdin = FakeStdin()
        self._returncode = None
        self.terminated = False

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = 0


class TestADBTransport(unittest.TestCase):
    def test_initialization_defaults(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport()
            self.assertEqual(transport.adb_binary, ADB_BINARY)
            self.assertEqual(transport.device_serial, DEVICE_SERIAL)
            self.assertTrue(transport.is_alive())
            transport.stop()

    def test_initialization_custom_params(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport(binary="custom_adb", device_serial="DEVICE_123")
            self.assertEqual(transport.adb_binary, "custom_adb")
            self.assertEqual(transport.device_serial, "DEVICE_123")
            expected_cmd = ["custom_adb", "-s", "DEVICE_123", "shell"]
            mock_popen.assert_called_once_with(
                expected_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            transport.stop()

    def test_no_accidental_second_process(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport(auto_start=True)
            self.assertEqual(mock_popen.call_count, 1)

            # Calling start again while running should NOT spawn a second process
            transport.start()
            transport.start()
            self.assertEqual(mock_popen.call_count, 1)

            # Sending commands should NOT spawn additional processes
            transport.send_tap(100, 200)
            transport.send_swipe(10, 20, 30, 40, 100)
            self.assertEqual(mock_popen.call_count, 1)
            transport.stop()

    def test_send_raw_command(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport()
            success = transport.send("echo hello")
            self.assertTrue(success)
            self.assertEqual(fake_proc.stdin.written, ["echo hello\n"])
            self.assertEqual(fake_proc.stdin.flushed, 1)
            transport.stop()

    def test_send_tap_formatting(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport()
            success = transport.send_tap(1280.4, 450.9)
            self.assertTrue(success)
            self.assertEqual(fake_proc.stdin.written, ["input tap 1280 450\n"])
            transport.stop()

    def test_send_swipe_formatting(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport()
            success = transport.send_swipe(100, 200, 300, 400, 150)
            self.assertTrue(success)
            self.assertEqual(fake_proc.stdin.written, ["input swipe 100 200 300 400 150\n"])
            transport.stop()

    def test_send_motionevent_formatting(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport()
            success = transport.send_motionevent("DOWN", 250, 480)
            self.assertTrue(success)
            self.assertEqual(fake_proc.stdin.written, ["input motionevent DOWN 250 480\n"])
            transport.stop()

    def test_stop_and_close_lifecycle(self):
        with patch("subprocess.Popen") as mock_popen:
            fake_proc = FakeProcess()
            mock_popen.return_value = fake_proc

            transport = ADBTransport()
            self.assertTrue(transport.is_alive())

            transport.stop()
            self.assertEqual(fake_proc.stdin.written, ["exit\n"])
            self.assertTrue(fake_proc.terminated)
            self.assertFalse(transport.is_alive())

            # Calling stop or close again should be safe and idempotent
            transport.close()
            self.assertFalse(transport.is_alive())

    def test_error_handling_broken_pipe_recovery(self):
        with patch("subprocess.Popen") as mock_popen:
            proc1 = FakeProcess()
            proc2 = FakeProcess()
            mock_popen.side_effect = [proc1, proc2]

            transport = ADBTransport()
            self.assertEqual(mock_popen.call_count, 1)

            # Simulate broken pipe on first write
            def broken_write(data):
                raise BrokenPipeError("Broken pipe")

            proc1.stdin.write = broken_write

            # send should trigger restart and retry
            success = transport.send("input tap 100 200")
            self.assertTrue(success)
            self.assertEqual(mock_popen.call_count, 2)
            self.assertEqual(proc2.stdin.written, ["input tap 100 200\n"])
            transport.stop()

    def test_error_handling_when_process_fails_to_start(self):
        with patch("subprocess.Popen", side_effect=OSError("ADB not found")):
            transport = ADBTransport(auto_start=False)
            transport.start()
            self.assertFalse(transport.is_alive())

            # Sending command should fail gracefully without unhandled exception
            result = transport.send("input tap 100 200")
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
