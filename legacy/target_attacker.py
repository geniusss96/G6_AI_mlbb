"""
Модуль авто-атаки по полоске здоровья (HP Bar) для Mobile Legends
================================================================
Вычисляет реальные координаты хитбокса героя на экране смартфона
на основе центра обнаруженной красной полоски ХП (класс hp_enemy от YOLOv8),
отображает прицел на экране и совершает мгновенное асинхронное нажатие (tap) 
через ADB в отдельном потоке без падения FPS видеопотока.
"""

import subprocess
import shutil
import os
import time
import queue
import threading
from typing import Tuple, Optional, List, Dict


# ==============================================================================
# КОНФИГУРАЦИЯ (PRODUCTION CONFIG)
# ==============================================================================

# Физическое разрешение экрана телефона в ландшафтном режиме (1544x720)
PHONE_WIDTH: int = 1544
PHONE_HEIGHT: int = 720

# Смещение вниз по оси Y в пикселях (под высоту экрана 720p)
Y_OFFSET_PHONE: int = 85

# Кулдаун между атаками в секундах (защита от перегрузки ADB и спама кликов)
# 0.20 = максимум 5 кликов в секунду (оптимально для базовой атаки/каста)
ATTACK_COOLDOWN: float = 0.20

# Серийный номер устройства ADB
DEVICE_SERIAL: Optional[str] = "R3CT90BBMTX"

# Имя бинарника ADB
ADB_BINARY: str = "adb"


class TargetAttacker:
    """
    Высокопроизводительный контроллер авто-прицеливания и атаки.
    Выполняет ADB tap в отдельном фоновом воркере, не блокируя поток OpenCV/YOLO.
    """

    def __init__(
        self,
        phone_width: int = PHONE_WIDTH,
        phone_height: int = PHONE_HEIGHT,
        y_offset_phone: int = Y_OFFSET_PHONE,
        cooldown: float = ATTACK_COOLDOWN,
        device_serial: Optional[str] = DEVICE_SERIAL,
        adb_binary: str = ADB_BINARY
    ):
        self.phone_w = phone_width
        self.phone_h = phone_height
        self.y_offset_phone = y_offset_phone
        self.cooldown = cooldown
        self.device_serial = device_serial
        self.adb_binary = adb_binary

        self.last_attack_time: float = 0.0

        # Очередь для асинхронных тапов (maxsize=1 для предотвращения задержек очереди)
        self._tap_queue: queue.Queue = queue.Queue(maxsize=1)
        self._find_adb_if_needed()

        # Запуск фонового воркера для отправки ADB-команд
        self._worker_thread = threading.Thread(target=self._adb_worker, daemon=True)
        self._worker_thread.start()

    def _find_adb_if_needed(self) -> None:
        """Автоматический поиск adb.exe в папке scrcpy или текущей директории."""
        if not shutil.which(self.adb_binary):
            candidates = [
                os.path.join(os.getcwd(), "scrcpy-win64-v4.1", "adb.exe"),
                os.path.join(os.getcwd(), "scrcpy", "adb.exe"),
                r"C:\Users\user\Desktop\scrcpy-win64-v4.1\adb.exe"
            ]
            for c in candidates:
                if os.path.isfile(c):
                    self.adb_binary = c
                    print(f"[*] TargetAttacker: Найден ADB -> {c}")
                    break

    def _adb_worker(self) -> None:
        """Фоновый поток, отправляющий тапы в ADB без задержки главного цикла."""
        while True:
            target = self._tap_queue.get()
            if target is None:
                break
            x, y = target
            cmd = [self.adb_binary]
            if self.device_serial:
                cmd.extend(["-s", self.device_serial])
            cmd.extend(["shell", "input", "tap", str(x), str(y)])

            try:
                subprocess.run(cmd, capture_output=True, timeout=1.0, check=False)
            except Exception:
                pass
            self._tap_queue.task_done()

    def select_best_target(
        self, 
        hp_boxes: List[Tuple[float, float, float, float]], 
        self_box: Optional[Tuple[float, float, float, float]],
        frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        """
        Выбирает самую приоритетную цель (ближайшего врага к нашему герою).
        Если наш герой не виден на кадре, берется цель, ближайшая к центру экрана.

        :param hp_boxes: Список центров полосок врагов [(x, y, w, h), ...]
        :param self_box: Координаты нашего героя (x, y, w, h) или None
        :param frame_shape: Размеры кадра (frame_h, frame_w)
        :return: (x_center, y_center) лучшей цели
        """
        if not hp_boxes:
            return None

        # Опорная точка: координаты нашего героя или центр экрана
        if self_box is not None:
            ref_x, ref_y = self_box[0], self_box[1]
        else:
            frame_h, frame_w = frame_shape
            ref_x, ref_y = frame_w / 2.0, frame_h / 2.0

        # Поиск полоски ХП с минимальным расстоянием до нашего персонажа
        best_target = None
        min_dist_sq = float('inf')

        for (x, y, w, h) in hp_boxes:
            dist_sq = (x - ref_x) ** 2 + (y - ref_y) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_target = (x, y)

        return best_target

    def process_and_attack(
        self,
        hp_x_frame: float,
        hp_y_frame: float,
        frame_w: int,
        frame_h: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Пересчитывает координаты и отправляет запрос на атаку с учетом кулдауна.

        :param hp_x_frame: Координата X центра полоски ХП на кадре
        :param hp_y_frame: Координата Y центра полоски ХП на кадре
        :param frame_w: Ширина кадра
        :param frame_h: Высота кадра
        :return: ((phone_x, phone_y), (frame_hitbox_x, frame_hitbox_y))
        """
        # 1. Масштабирование координат кадра в телефон (пропорция)
        scale_x = self.phone_w / float(frame_w)
        scale_y = self.phone_h / float(frame_h)

        phone_hp_x = int(round(hp_x_frame * scale_x))
        phone_hp_y = int(round(hp_y_frame * scale_y))

        # 2. Смещение на хитбокс в координатах телефона
        phone_target_x = max(0, min(phone_hp_x, self.phone_w - 1))
        phone_target_y = max(0, min(phone_hp_y + self.y_offset_phone, self.phone_h - 1))

        # 3. Вычисление координаты хитбокса на исходном кадре (для отрисовки прицела в OpenCV)
        frame_y_offset = int(round(self.y_offset_phone / scale_y))
        frame_hitbox_x = int(round(hp_x_frame))
        frame_hitbox_y = min(frame_h - 1, int(round(hp_y_frame + frame_y_offset)))

        # 4. Проверка кулдауна и отправка тапа в неблокирующую очередь
        current_time = time.time()
        if (current_time - self.last_attack_time) >= self.cooldown:
            # Очищаем старую невыполненную задачу, если очередь занята
            if not self._tap_queue.empty():
                try:
                    self._tap_queue.get_nowait()
                except queue.Empty:
                    pass
            self._tap_queue.put((phone_target_x, phone_target_y))
            self.last_attack_time = current_time

        return (phone_target_x, phone_target_y), (frame_hitbox_x, frame_hitbox_y)
