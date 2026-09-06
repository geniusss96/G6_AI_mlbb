"""
control.adb — Centralized low-level ADB transport layer for Claude Tactical AI V2.

Provides persistent interactive ADB shell communication with <0.1ms dispatch latency.
Eliminates overhead of spawning new subprocesses per tap/swipe.
Coordinates neutral: accepts raw device coordinates and commands without knowing about
game buttons, joystick center, or hero skills.
"""

from typing import Optional, List
import subprocess
import shutil
import os
import threading
import atexit

from config.config import ADB_BINARY, DEVICE_SERIAL


class ADBTransport:
    """
    Low-level centralized ADB transport.
    Maintains a single persistent interactive shell session to Android over USB/TCP.
    """

    def __init__(
        self,
        binary: Optional[str] = None,
        device_serial: Optional[str] = None,
        auto_start: bool = True,
    ):
        if binary is None:
            self.adb_binary = ADB_BINARY
            self._find_adb_binary_if_needed()
        else:
            self.adb_binary = binary
        self.device_serial = device_serial or DEVICE_SERIAL

        self.base_cmd = [self.adb_binary]
        if self.device_serial:
            self.base_cmd.extend(["-s", self.device_serial])

        self._shell_proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()

        if auto_start:
            self.start()

        atexit.register(self.stop)

    def _find_adb_binary_if_needed(self) -> None:
        """Locates adb executable in workspace or system PATH if default binary not found."""
        if not shutil.which(self.adb_binary):
            candidates = [
                os.path.join(os.getcwd(), "adb.exe"),
                os.path.join(os.getcwd(), "adb"),
                os.path.join(os.getcwd(), "scrcpy-win64-v4.1", "adb.exe"),
                os.path.join(os.getcwd(), "scrcpy", "adb.exe"),
                r"c:\Users\user\Desktop\platform-tools\adb.exe",
                r"c:\Users\user\Desktop\scrcpy-win64-v4.1\adb.exe",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    self.adb_binary = c
                    break

    def _start_unlocked(self, force: bool = False) -> None:
        """Internal spawn without acquiring lock (caller must hold _lock)."""
        if not force and self._shell_proc and self._shell_proc.poll() is None:
            return

        if self._shell_proc:
            try:
                self._shell_proc.terminate()
            except Exception:
                pass
            self._shell_proc = None

        cmd = list(self.base_cmd) + ["shell"]
        try:
            self._shell_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
        except Exception as e:
            # Retain V1 behavior: warn and proceed gracefully
            print(f"[!] Warning: ADBTransport failed to launch interactive shell: {e}")
            self._shell_proc = None

    def start(self) -> None:
        """Starts or restarts the persistent interactive ADB shell."""
        with self._lock:
            self._start_unlocked()

    def send(self, command: str) -> bool:
        """
        Sends a single raw shell command string into the persistent interactive ADB shell.

        Args:
            command: Shell command (e.g. 'input tap 500 300' or 'input motionevent MOVE ...')

        Returns:
            True if sent successfully, False otherwise.
        """
        with self._lock:
            if not self._shell_proc or self._shell_proc.poll() is not None:
                self._start_unlocked()

            if not self._shell_proc or not self._shell_proc.stdin:
                return False

            try:
                self._shell_proc.stdin.write(command.strip() + "\n")
                self._shell_proc.stdin.flush()
                return True
            except Exception:
                # Broken pipe, closed pipe, or disconnected: restart shell and retry once (V1 parity)
                self._start_unlocked(force=True)
                if self._shell_proc and self._shell_proc.stdin:
                    try:
                        self._shell_proc.stdin.write(command.strip() + "\n")
                        self._shell_proc.stdin.flush()
                        return True
                    except Exception:
                        return False
                return False

    def send_tap(self, x: int, y: int) -> bool:
        """Formats and sends an 'input tap X Y' command."""
        return self.send(f"input tap {int(x)} {int(y)}")

    def send_swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 100,
    ) -> bool:
        """Formats and sends an 'input swipe X1 Y1 X2 Y2 DURATION' command."""
        return self.send(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}")

    def send_motionevent(self, action: str, x: int, y: int) -> bool:
        """
        Formats and sends an 'input motionevent ACTION X Y' command.
        Action is typically 'DOWN', 'MOVE', or 'UP'.
        """
        return self.send(f"input motionevent {action} {int(x)} {int(y)}")

    def is_alive(self) -> bool:
        """Checks if persistent shell process is currently running."""
        with self._lock:
            return self._shell_proc is not None and self._shell_proc.poll() is None

    def stop(self) -> None:
        """Terminates the persistent ADB shell process cleanly."""
        with self._lock:
            if self._shell_proc:
                try:
                    if self._shell_proc.stdin:
                        try:
                            self._shell_proc.stdin.write("exit\n")
                            self._shell_proc.stdin.flush()
                        except Exception:
                            pass
                    self._shell_proc.terminate()
                except Exception:
                    pass
                finally:
                    self._shell_proc = None

    def close(self) -> None:
        """Alias for stop() for parity with V1 controller interfaces."""
        self.stop()
