"""
Модуль управления виртуальным джойстиком перемещения (Movement Joystick)
========================================================================
Позволяет персонажу двигаться в направлении цели (врага), кайтить (отходить)
или перемещаться в заданном направлении, управляя левым экранным стиком в MLBB.

Калибровка под физический экран смартфона 720x1544 (Landscape, rotation 3):
  Центр стика в ландшафте: X = 288, Y = 560
  Радиус: 110 px
"""

import math
import time
import subprocess
import shutil
import os
import atexit
import threading
from typing import Tuple, Optional
from config.config import (
    JOYSTICK_CENTER_X,
    JOYSTICK_CENTER_Y,
    JOYSTICK_RADIUS,
    ADB_BINARY,
    DEVICE_SERIAL,
)


class JoystickController:
    # --------------------------------------------------------------------------
    # Точные координаты центра джойстика в альбомной ориентации 1544x720
    # --------------------------------------------------------------------------
    JOYSTICK_CENTER_X: int = JOYSTICK_CENTER_X
    JOYSTICK_CENTER_Y: int = JOYSTICK_CENTER_Y
    JOYSTICK_RADIUS: int = JOYSTICK_RADIUS

    def __init__(
        self,
        center_x: int = JOYSTICK_CENTER_X,
        center_y: int = JOYSTICK_CENTER_Y,
        radius: int = JOYSTICK_RADIUS,
        adb_binary: str = ADB_BINARY,
        device_serial: Optional[str] = DEVICE_SERIAL
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.adb_binary = adb_binary
        self.device_serial = device_serial

        self._find_adb_if_needed()

        self.is_holding = False
        self.current_pos = (self.center_x, self.center_y)
        self._shell_proc = None
        self._lock = threading.Lock()
        self._init_shell()

        atexit.register(self.release)

    def _find_adb_if_needed(self) -> None:
        """Автоматический поиск adb.exe в папке scrcpy или текущей директории."""
        if not shutil.which(self.adb_binary):
            candidates = [
                os.path.join(os.getcwd(), "scrcpy-win64-v4.1", "adb.exe"),
                os.path.join(os.getcwd(), "scrcpy", "adb.exe"),
                r"c:\Users\user\Desktop\platform-tools\adb.exe",
                r"c:\Users\user\Desktop\scrcpy-win64-v4.1\adb.exe",
                os.path.join(os.getcwd(), "adb.exe")
            ]
            for c in candidates:
                if os.path.isfile(c):
                    self.adb_binary = c
                    break

    def _init_shell(self) -> None:
        """Запускает постоянный низколатентный сеанс ADB shell."""
        cmd = [self.adb_binary]
        if self.device_serial:
            cmd.extend(["-s", self.device_serial])
        cmd.extend(["shell"])
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
            print(f"[!] Ошибка запуска интерактивного ADB shell: {e}")

    def _send_cmd(self, line: str) -> None:
        """Отправляет команду напрямую в stdin открытого ADB shell (0.1 мс задержки)."""
        with self._lock:
            if not self._shell_proc or self._shell_proc.poll() is not None:
                self._init_shell()
            try:
                self._shell_proc.stdin.write(line + "\n")
                self._shell_proc.stdin.flush()
            except Exception:
                self._init_shell()
                try:
                    self._shell_proc.stdin.write(line + "\n")
                    self._shell_proc.stdin.flush()
                except Exception:
                    pass

        self.last_down_time = 0.0

    def notify_screen_touched(self) -> None:
        """
        Уведомляет джойстик о том, что был произведен тап/каст по экрану (атака, скилл).
        Так как утилита 'input' в Android сбрасывает касания при выполнении тапа/свайпа,
        сброс флага is_holding заставляет джойстик на следующем кадре мгновенно 
        послать новый 'input motionevent DOWN', предотвращая паралич и остановку бега.
        """
        self.is_holding = False

    def send_raw_cmd(self, line: str) -> None:
        """
        Выполняет команду напрямую в открытом низколатентном потоке ADB shell.
        Используется claude_macro для сверхбыстрых тапов без оверхеда subprocess (0.1мс).
        """
        self._send_cmd(line)

    def move_towards(
        self,
        hero_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        duration_ms: int = 400
    ) -> Tuple[int, int]:
        """
        Непрерывное ведение джойстика с автоматическим восстановлением контакта (motionevent).
        Персонаж бежит плавно, не зависая при стрельбе или касте скиллов.
        """
        hx, hy = hero_pos
        tx, ty = target_pos

        dx = tx - hx
        dy = ty - hy
        angle = math.atan2(dy, dx)

        target_joy_x = int(round(self.center_x + self.radius * math.cos(angle)))
        target_joy_y = int(round(self.center_y + self.radius * math.sin(angle)))

        self._toggle = not getattr(self, '_toggle', False)
        jitter_x = target_joy_x + (1 if self._toggle else -1)

        now = time.time()
        # Если контакт потерян или удерживается дольше 1.8с (защита от засыпания драйвера)
        if not self.is_holding or (now - self.last_down_time > 1.8):
            self._send_cmd(f"input motionevent DOWN {self.center_x} {self.center_y}")
            self._send_cmd(f"input motionevent MOVE {target_joy_x} {target_joy_y}")
            self.is_holding = True
            self.last_down_time = now
        else:
            self._send_cmd(f"input motionevent MOVE {jitter_x} {target_joy_y}")

        self.current_pos = (target_joy_x, target_joy_y)
        return target_joy_x, target_joy_y

    def kite_away(
        self,
        hero_pos: Tuple[float, float],
        enemy_pos: Tuple[float, float],
        duration_ms: int = 400
    ) -> Tuple[int, int]:
        """
        Непрерывный отход от врага (кайтинг) с автоматическим восстановлением контакта.
        """
        hx, hy = hero_pos
        ex, ey = enemy_pos

        dx = hx - ex
        dy = hy - ey
        angle = math.atan2(dy, dx)

        target_joy_x = int(round(self.center_x + self.radius * math.cos(angle)))
        target_joy_y = int(round(self.center_y + self.radius * math.sin(angle)))

        self._toggle = not getattr(self, '_toggle', False)
        jitter_x = target_joy_x + (1 if self._toggle else -1)

        now = time.time()
        if not self.is_holding or (now - self.last_down_time > 1.8):
            self._send_cmd(f"input motionevent DOWN {self.center_x} {self.center_y}")
            self._send_cmd(f"input motionevent MOVE {target_joy_x} {target_joy_y}")
            self.is_holding = True
            self.last_down_time = now
        else:
            self._send_cmd(f"input motionevent MOVE {jitter_x} {target_joy_y}")

        self.current_pos = (target_joy_x, target_joy_y)
        return target_joy_x, target_joy_y

    def release(self) -> None:
        """
        Отпускает виртуальный джойстик (палец поднимается с экрана, герой останавливается).
        """
        if self.is_holding:
            cx, cy = self.current_pos
            self._send_cmd(f"input motionevent UP {cx} {cy}")
            self.is_holding = False
            self.current_pos = (self.center_x, self.center_y)

    def close(self) -> None:
        """Безопасное завершение работы контроллера."""
        self.release()
        if self._shell_proc:
            try:
                self._shell_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    print("==================================================")
    print("   MLBB Joystick Landscape Calibration (1544x720) ")
    print("==================================================")
    joystick = JoystickController()
    print(f"[*] Center: ({joystick.center_x}, {joystick.center_y})")
    print(f"[*] Radius: {joystick.radius} px")
    joystick.close()
