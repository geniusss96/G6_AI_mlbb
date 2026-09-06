"""
Тактический ИИ-ассистент Клода (Claude Tactical AI Brain + Q-Learning Engine)
========================================================================
Интеллектуальная система принятия решений и обучения с подкреплением:

Возможности:
  1. [Q-LEARNING ROAMING] - Автономное обучение маршрутам разведки (линия, река, лес, фланги).
  2. [BRUTAL DEATH PENALTY] - Жесткое наказание за смерть (-50% всех морковок, минимум 200).
  3. [100% ANTI-CORPSE]   - Полная ликвидация стрельбы по мертвому Зилонгу:
                            регистрация трупа до оптики, фильтр монолитности красного плаща,
                            сторожевой таймер неподвижных объектов (>3.5с -> черный список).
  4. [HP-AWARENESS]       - Сенсор своего здоровья (Паника при <32% HP, Хищник при преимуществе).
  5. [AUTO LEVEL-UP & REGEN] - Авто-прокачка навыков по плюсикам и своевременный хил.
  6. [ANTI-WALL WATCHDOG] - Детектор утыкания в стены с отскоком и штрафом в Q-памяти.
  7. [AI MOOD & K/D HUD]  - Живой боевой дашборд настроения ИИ, K/D статистики и Q-весов.
"""

import time
import os
import sys
import ctypes
import cv2
import numpy as np
import mss
import win32gui
import win32con
import win32api
from ultralytics import YOLO

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from claude_macro import ClaudeMacroController
from joystick_controller import JoystickController
from claude_brain import ClaudeRLBrain
from vision.capture import ScreenCapture
from vision.detector import YoloDetector, Detection
from vision.skill_state import SkillStateChecker
from vision.hp_detector import HPDetector, HPObservation


from config.config import (
    WEIGHTS_PATH,
    CONF_HERO_THRESHOLD,
    CONF_MINION_THRESHOLD,
    CONF_BUFF_THRESHOLD,
    CONF_TURRET_THRESHOLD,
    YOLO_DEVICE,
    YOLO_IMGSZ,
    DEVICE_SERIAL,
    SHOW_PREVIEW_WINDOW,
    PREVIEW_WIDTH,
    PREVIEW_HEIGHT,
    KITING_DANGER_DIST_PX,
    SWEET_SPOT_MAX_DIST_PX,
    TURRET_DANGER_DIST_PX,
    JOYSTICK_STEP_SEC,
    BASIC_ATTACK_COOLDOWN_SEC,
    COMBO_COOLDOWN_SEC,
    S1_COOLDOWN_SEC,
    S2_COOLDOWN_SEC,
    ULT_COOLDOWN_SEC,
    LEVELUP_INTERVAL_SEC,
    REGEN_COOLDOWN_SEC,
    ROAM_LEG_DURATION,
    IDLE_PENALTY_INTERVAL,
    SKILL_CHECK_COORDS,
)

# ============================================================================
# ДЕТЕКТОР ГОТОВНОСТИ СПОСОБНОСТЕЙ (Делегирование в vision.skill_state)
# ============================================================================
_skill_checker = SkillStateChecker()


# ============================================================================
# СИСТЕМНЫЕ ФУНКЦИИ ОКНА SCRCPY (Делегирование в vision.capture)
# ============================================================================
_screen_capture = ScreenCapture()


def find_scrcpy_window():
    return _screen_capture.find_scrcpy_window()


def get_client_area(hwnd):
    return _screen_capture.get_client_area(hwnd)


def is_valid_battlefield_target(x_center: float, y_center: float, box_w: float, box_h: float, frame_w: int, frame_h: int) -> bool:
    """
    Интеллектуальный фильтр интерфейса (HUD Exclusion Mask):
    Отсекает ложные срабатывания нейросети на 2D элементах интерфейса:
      1. Верхняя панель счета/времени и полоска выбранной цели (y < 18% высоты)
      2. Миникарта в верхнем левом углу (x < 22%, y < 40%)
      3. Кнопки способностей и атаки в правом нижнем углу (x > 78%, y > 65%)
      4. Аномально гигантские рамки босс-баров или фонового мусора
    """
    if (y_center / frame_h) < 0.18:
        return False
    if (x_center / frame_w < 0.22) and (y_center / frame_h < 0.40):
        return False
    if (x_center / frame_w > 0.78) and (y_center / frame_h > 0.65):
        return False
    if box_w > 320 or box_h > 120 or box_w < 15:
        return False
    return True


_hp_detector = HPDetector()


def detect_red_hp_bars(frame: np.ndarray) -> list:
    return _hp_detector.detect_red_hp_bars(frame)


def is_real_enemy_hp_bar(frame: np.ndarray, xywh: list) -> bool:
    return _hp_detector.is_real_enemy_hp_bar(frame, xywh)


def is_real_enemy_minion(frame: np.ndarray, xywh: list, is_near_base: bool) -> bool:
    return _hp_detector.is_real_enemy_minion(frame, xywh, is_near_base)


def get_claude_hp_ratio(frame: np.ndarray, self_target: tuple) -> float:
    return _hp_detector.detect_player_hp(frame, self_target).value


# ============================================================================
# СИСТЕМА ПООЩРЕНИЙ И ОБУЧЕНИЯ С ПОДКРЕПЛЕНИЕМ (CARROT REWARD ENGINE)
# ============================================================================
class CarrotRewardSystem:
    """
    Система похвалы и подкрепления (Reinforcement Learning Rewards):
    Награждает бота 'морковками' (Carrots) за реальные боевые достижения и жестко наказывает:
      +50 🥕 - Убийство вражеского героя (Зилонг повержен!)
      +30 🥕 - Успешное прокастованное комбо
      +15 🥕 - Обнаружение врага при разведке
      +10 🥕 - Убийство крипа/монстра
      💀 СМЕРТЬ: Отбирается 50% ВСЕХ накопленных морковок (минимум 200) и падает уровень!
    """
    def __init__(self):
        self.carrots = 0
        self.level = 1
        self.kills = 0
        self.deaths = 0
        self.highlights_count = 0
        self.last_event_text = "READY FOR SCOUTING"
        self.last_event_time = 0.0
        self.last_scout_reward_time = 0.0
        self.enemy_was_visible = False

    def add_reward(self, amount: int, reason: str):
        self.carrots = max(0, self.carrots + amount)
        self.last_event_text = f"+{amount} CARROTS: {reason}"
        self.last_event_time = time.time()
        self.level = max(1, 1 + (self.carrots // 100))
        print(f"\n[🥕 НАГРАДА] {self.last_event_text} (Всего морковок: {self.carrots} | LVL {self.level})")

    def penalize_death(self) -> int:
        """ЖЕСТКОЕ НАКАЗАНИЕ: отбирает 50% всех морковок (минимум 200) и роняет ранг."""
        self.deaths += 1
        loss = max(200, int(self.carrots * 0.50))
        prev = self.carrots
        self.carrots = max(0, self.carrots - loss)
        deducted = prev - self.carrots
        self.level = max(1, 1 + (self.carrots // 100))
        self.last_event_text = f"💀 DEATH! -{deducted} CARROTS (-50%)"
        self.last_event_time = time.time()
        print(f"\n[💀 ЖЕСТОЧАЙШЕЕ НАКАЗАНИЕ] Клод погиб! Отобрано {deducted} морковок (50%)! Осталось: {self.carrots} | Уровень упал до: {self.level}")
        return deducted

    def penalize_idle(self, amount: int = 15, reason: str = "IDLE (NO ENEMIES)") -> int:
        """Штраф за пустое шатание по карте без нахождения врагов."""
        prev = self.carrots
        self.carrots = max(0, self.carrots - amount)
        deducted = prev - self.carrots
        self.level = max(1, 1 + (self.carrots // 100))
        self.last_event_text = f"⏱️ IDLE: -{deducted} CARROTS (SEARCHING!)"
        self.last_event_time = time.time()
        print(f"\n[⏱️ ШТРАФ ЗА БЕЗДЕЙСТВИЕ] Долго бродит без пользы! -{deducted} морковок. Баланс: {self.carrots} | LVL {self.level}")
        return deducted

    def render_hud(self, frame, mood_text: str, mood_color: tuple, q_info: str):
        h, w = frame.shape[:2]
        # Плашка наград и аналитики в нижнем левом углу
        cv2.rectangle(frame, (10, h - 92), (430, h - 6), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, h - 92), (430, h - 6), (0, 140, 255), 2)

        # Строка 1: Морковки, Уровень, K/D и Сохраненные лучшие моменты
        cv2.putText(frame, f"CARROTS: {self.carrots} (LVL {self.level}) | K/D: {self.kills}/{self.deaths} | BEST: {self.highlights_count}",
                    (18, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)

        # Строка 2: Настроение ИИ и статус
        cv2.putText(frame, f"MOOD: {mood_text}",
                    (18, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mood_color, 2)

        # Строка 3: Последнее событие или инфо о Q-обучении
        now = time.time()
        if (now - self.last_event_time) < 4.0:
            if "DEATH" in self.last_event_text:
                color = (0, 0, 255)
            elif "🏆" in self.last_event_text or "HERO SLAIN" in self.last_event_text:
                color = (0, 215, 255)  # Золотой цвет для триумфальных моментов
            else:
                color = (0, 255, 0)
            cv2.putText(frame, self.last_event_text, (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        else:
            cv2.putText(frame, f"Q-BRAIN: {q_info}", (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


# ============================================================================
# ГЛАВНЫЙ ЦИКЛ БОТА С ТАКТИЧЕСКИМ ИНТЕЛЛЕКТОМ И Q-LEARNING
# ============================================================================
def main():
    print("==================================================")
    print("    CLAUDE AUTONOMOUS Q-LEARNING TACTICAL BRAIN   ")
    print("==================================================")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] Не найден файл весов:\n{WEIGHTS_PATH}")
        return

    print(f"[*] Загружаем нейросеть: {WEIGHTS_PATH}...")
    detector = YoloDetector(weights_path=WEIGHTS_PATH)
    class_map = detector.class_map
    hp_enemy_id = class_map.get("hp_enemy", 1)
    hp_self_id = class_map.get("hp_self", 0)
    turret_enemy_id = class_map.get("turret_enemy", 4)
    minion_class_id = class_map.get("minion_enemy", 2)
    minion_ally_id = class_map.get("minion_ally", 3)
    bush_class_id = class_map.get("bush", 6)
    buff_class_ids = set([class_map[n] for n in ["red_baff", "blue_baff", "money_crab", "vazon", "money"] if n in class_map])
    tron_self_id = class_map.get("tron_self", 12)

    print(f"[*] Инициализация контроллеров Клода и Q-Brain...")
    try:
        claude_macro = ClaudeMacroController(serial=DEVICE_SERIAL)
    except Exception as e:
        print(f"[!] Предупреждение ADB: {e}")
        claude_macro = None

    joystick = JoystickController(device_serial=DEVICE_SERIAL)
    if claude_macro:
        claude_macro.set_joystick(joystick)
    reward_sys = CarrotRewardSystem()
    rl_brain = ClaudeRLBrain(memory_file="q_brain.json")

    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)

    cv_window_name = "Claude Tactical AI"
    if SHOW_PREVIEW_WINDOW:
        cv2.namedWindow(cv_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv_window_name, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        print("[*] ВИЗУАЛЬНОЕ ОКНО ИИ: Активировано (аппаратная защита WDA_EXCLUDEFROMCAPTURE включена)")
    else:
        print("[*] РЕЖИМ ЧИСТОГО ЭКРАНА: Включен")

    window_affinity_set = False
    last_joystick_time = 0.0
    last_attack_time = 0.0
    last_s1_time = 0.0
    last_s2_time = 0.0
    last_combo_time = 0.0
    last_skill_levelup_time = 0.0
    last_regen_time = 0.0
    frame_count = 0

    cached_detections: list[Detection] = []
    corpse_zones = []  # list of {"pos": (x, y), "expiry": timestamp}
    last_known_enemy_time = 0.0
    last_attacked_enemy_pos = None
    last_attacked_enemy_time = 0.0
    last_farm_pos = None

    # Трекеры целей с удержанием (Target Persistence Hysteresis)
    tracked_enemy_pos = None
    tracked_enemy_time = 0.0
    tracked_farm_pos = None
    tracked_farm_time = 0.0

    stationary_target_pos = None
    stationary_target_time = 0.0

    enemy_disappeared_since = 0.0
    farm_attack_start_time = 0.0
    farm_disappeared_since = 0.0

    current_roam_action = None
    current_roam_target = None
    roam_leg_start_time = 0.0
    ROAM_LEG_DURATION = 3.0  # Удерживать направление 3.0 секунды перед сменой (активный бег)

    last_combat_or_farm_time = time.time()
    last_combat_engagement_time = 0.0
    IDLE_PENALTY_INTERVAL = 25.0  # Разумный интервал: дает дойти до линии и исследовать сектор без штрафов

    last_self_seen_time = time.time()
    last_known_hp_ratio = 1.0
    is_claude_dead = False
    claude_death_time = 0.0
    combat_stance = "SWEET_SPOT"

    # Режим быстрого выхода на линию (Rush to Lane Highway):
    # Первые 25 секунд матча - непрерывный спринт на линию к внешним вышкам!
    match_start_time = time.time()
    rush_to_lane_end_time = match_start_time + 25.0
    last_quick_buy_time = 0.0

    reward_sys.highlights_count = len(rl_brain.best_moments)

    print("\n[АВТОНОМНЫЙ РЕЖИМ ИИ АКТИВИРОВАН]:")
    print("  - Q-Learning автономное исследование карты (6 стратегических векторов)")
    print("  - Жесткое наказание за смерть (-50% морковок, минимум 200)")
    print("  - 100% защита от трупа Зилонга (монолитность контура + Anti-Stuck watchdog)")
    print("  - Сенсор здоровья (Паника при <32% HP, Хищник при преимуществе)")
    print("  - Авто-прокачка способностей и тактический авто-хил\n")

    try:
        with ScreenCapture() as capture:
            while True:
                start_time = time.time()

                # 1. Захват окна scrcpy и нормализация кадра в 1544x720 через vision.capture
                frame = capture.grab()
                if frame is None:
                    hwnd = capture.find_scrcpy_window()
                    if not hwnd:
                        print(f"\r[!] Окно scrcpy не найдено. Ожидание...", end="", flush=True)
                        time.sleep(0.5)
                    else:
                        time.sleep(0.1)
                    continue

                frame_h, frame_w = frame.shape[:2]

                now = time.time()

                # Авто-прокачка скиллов раз в 25 секунд (без спама кликами)
                if claude_macro and (now - last_skill_levelup_time) >= LEVELUP_INTERVAL_SEC:
                    claude_macro.level_up_skills()
                    last_skill_levelup_time = now

                # 2. Сенсор способностей со строгим таймером кулдаунов (Anti-Spam Guard)
                skill_state = _skill_checker.check(frame)
                s1_ok = skill_state.s1_ready and ((now - last_s1_time) >= S1_COOLDOWN_SEC)
                s2_ok = skill_state.s2_ready and ((now - last_s2_time) >= S2_COOLDOWN_SEC)
                ult_ok = skill_state.ultimate_ready and ((now - last_combo_time) >= ULT_COOLDOWN_SEC)
                can_full_combo = s2_ok and ult_ok

                # 3. Инференс нейросети YOLOv8 через vision.detector (FP16 с пропуском через кадр)
                if frame_count % 2 == 0 or not cached_detections:
                    cached_detections = detector.detect(
                        frame=frame,
                        conf=min(CONF_HERO_THRESHOLD, CONF_BUFF_THRESHOLD),
                        half=True
                    )

                annotated_frame = frame.copy()

                # Очистка устаревших зон трупов (по истечении времени)
                corpse_zones = [cz for cz in corpse_zones if now < cz["expiry"]]

                # 4. Сбор объектов на поле боя от YOLO
                yolo_enemy_heroes = []
                raw_enemy_minions = []
                ally_minions = []
                farm_targets = []
                enemy_turrets = []
                detected_bushes = []
                self_target = None
                tron_self_detected = False

                for det in cached_detections:
                    cls_id = det.class_id
                    conf = det.confidence
                    xywh = det.xywh

                    # Фильтр интерфейса (HUD Exclusion Mask)
                    if not is_valid_battlefield_target(xywh[0], xywh[1], xywh[2], xywh[3], frame_w, frame_h):
                        continue

                    if cls_id == hp_enemy_id and conf >= CONF_HERO_THRESHOLD:
                        # 100% ЦВЕТОВОЙ ФИЛЬТР: проверяем, что полоска РЕАЛЬНО КРАСНАЯ (не союзник!)
                        if is_real_enemy_hp_bar(frame, xywh):
                            yolo_enemy_heroes.append(xywh)
                    elif cls_id == hp_self_id and conf >= 0.35:
                        self_target = (xywh[0], xywh[1])
                    elif cls_id == turret_enemy_id and conf >= CONF_TURRET_THRESHOLD:
                        enemy_turrets.append(xywh)
                    elif cls_id == tron_self_id and conf >= 0.30:
                        tron_self_detected = True
                    elif cls_id == minion_ally_id and conf >= 0.22:
                        ally_minions.append(xywh)
                    elif cls_id == minion_class_id and conf >= CONF_MINION_THRESHOLD:
                        raw_enemy_minions.append((xywh, conf))
                    elif cls_id in buff_class_ids and conf >= CONF_BUFF_THRESHOLD:
                        farm_targets.append(xywh)
                    elif cls_id == bush_class_id and conf >= 0.35:
                        detected_bushes.append(xywh)

                # Детектор нахождения на базе (стартовый спринт или трон базы на экране)
                is_near_base = (now < rush_to_lane_end_time) or (tron_self_detected and (now - match_start_time) < 240.0)

                # Строгий фильтр вражеских крипов: отсекаем союзных крипов, зону базы и черные списки
                for m_box, m_conf in raw_enemy_minions:
                    if not is_real_enemy_minion(frame, m_box, is_near_base):
                        continue
                    # Не совпадает ли с союзным крипом (дистанция < 65px)?
                    is_ally = any(
                        (m_box[0] - ax)**2 + (m_box[1] - ay)**2 < 65**2
                        for ax, ay, _, _ in ally_minions
                    )
                    # Не находится ли в черном списке неуязвимых/трупов?
                    is_blacklisted = any(
                        (m_box[0] - cz["pos"][0])**2 + (m_box[1] - cz["pos"][1])**2 < 60**2
                        for cz in corpse_zones
                    )
                    if not is_ally and not is_blacklisted:
                        farm_targets.append(m_box)

                # Позиция нашего Клода
                ref_pt = self_target if self_target else (frame_w / 2.0, frame_h / 2.0)

                # Оценка процента своего здоровья (HP-Awareness)
                hp_ratio = get_claude_hp_ratio(frame, self_target)

                # 5. Оптический сенсор полосок здоровья врагов (ТОЛЬКО вне базы)
                enemy_heroes = list(yolo_enemy_heroes)
                if not is_near_base:
                    optical_red_bars = detect_red_hp_bars(frame)
                    for r_box in optical_red_bars:
                        rx, ry, rw, rh = r_box
                        if not any((rx - ex)**2 + (ry - ey)**2 < 50**2 for ex, ey, _, _ in enemy_heroes):
                            enemy_heroes.append(r_box)

                # Вычисляем опасность вражеской вышки
                turret_danger = False
                if enemy_turrets:
                    enemy_turrets.sort(key=lambda t: (t[0] - ref_pt[0])**2 + (t[1] - ref_pt[1])**2)
                    dist_turret = np.sqrt((enemy_turrets[0][0] - ref_pt[0])**2 + (enemy_turrets[0][1] - ref_pt[1])**2)
                    if dist_turret < TURRET_DANGER_DIST_PX:
                        turret_danger = True

                # Вычисляем ближайшего врага с удержанием цели (Target Persistence Hysteresis)
                best_enemy = None
                dist_enemy = 9999.0
                if enemy_heroes:
                    enemy_heroes.sort(key=lambda t: (t[0] - ref_pt[0])**2 + (t[1] - ref_pt[1])**2)
                    tracked_enemy_pos = (enemy_heroes[0][0], enemy_heroes[0][1])
                    tracked_enemy_time = now
                    best_enemy = tracked_enemy_pos
                    dist_enemy = np.sqrt((best_enemy[0] - ref_pt[0])**2 + (best_enemy[1] - ref_pt[1])**2)
                    last_known_enemy_time = now
                    last_combat_engagement_time = now
                elif tracked_enemy_pos and (now - tracked_enemy_time) < 0.90:
                    # Удержание цели: если враг моргнул на 1-2 кадра, НЕ теряем его!
                    best_enemy = tracked_enemy_pos
                    dist_enemy = np.sqrt((best_enemy[0] - ref_pt[0])**2 + (best_enemy[1] - ref_pt[1])**2)
                else:
                    best_enemy = None
                    tracked_enemy_pos = None

                # НАДЕЖНАЯ ПРОВЕРКА СМЕРТИ ВРАГА (БЕЗ ЛОЖНЫХ СРАБАТЫВАНИЙ):
                # Враг повержен, если шел активный бой (>=2.0с) и он отсутствует более 2.5 секунд
                if enemy_heroes:
                    enemy_disappeared_since = 0.0
                else:
                    if last_attacked_enemy_pos:
                        if enemy_disappeared_since == 0.0:
                            enemy_disappeared_since = now
                        elif (now - enemy_disappeared_since) >= 2.5:
                            # Убийство засчитывается, если бой длился >=2.0с и последняя атака была свежей
                            if (now - last_attacked_enemy_time) <= 3.5 and (now - last_combat_engagement_time) >= 2.0:
                                reward_sys.kills += 1
                                reward_sys.add_reward(50, "ENEMY HERO SLAIN (ZILONG DEAD)!")
                                rl_brain.reward_combat(60.0, "KILLED ENEMY HERO")
                                rl_brain.reward_roam(40.0, "ROUTE LED TO KILL")
                                # Заносим триумф в «Золотой фонд» памяти ИИ!
                                rl_brain.record_best_moment(
                                    moment_type="HERO_KILL",
                                    reward=80.0,
                                    description="Соло-ликвидация вражеского героя (Зилонг повержен!)",
                                    hp_left_pct=int(hp_ratio * 100)
                                )
                                reward_sys.highlights_count = len(rl_brain.best_moments)
                            last_attacked_enemy_pos = None
                            enemy_disappeared_since = 0.0

                # Отслеживание гибели нашего героя Клода:
                # Клод может погибнуть ТОЛЬКО в активном бою или под вышкой при низком HP!
                in_combat_recently = ((now - last_combat_engagement_time) < 4.0) or turret_danger
                if self_target:
                    last_self_seen_time = now
                    last_known_hp_ratio = hp_ratio
                    if is_claude_dead:
                        is_claude_dead = False
                        rush_to_lane_end_time = now + 14.0
                        print("\n[⚔️ ВОЗРОЖДЕНИЕ] Клод снова в строю! Активирован марш на линию (RUSH TO LANE)!")
                else:
                    # Авто-возрождение по таймеру тренировочного режима (6 секунд)
                    if is_claude_dead and (now - claude_death_time) > 6.0:
                        is_claude_dead = False
                        rush_to_lane_end_time = now + 14.0
                        print("\n[⚔️ ТАЙМЕР РЕСПАУНА ИСТЕК] Клод возродился и идет в бой! Активирован марш на линию (RUSH TO LANE)!")

                    if (now - last_self_seen_time > 2.5) and not is_claude_dead:
                        if in_combat_recently and last_known_hp_ratio < 0.28:
                            is_claude_dead = True
                            claude_death_time = now
                            reward_sys.penalize_death()
                            rl_brain.reward_roam(-30.0, "DIED ON ROUTE")
                            rl_brain.reward_combat(-40.0, "DIED IN COMBAT")
                        else:
                            # Вне боя отсутствие полоски HP означает 100% здоровье (скрыто игрой)
                            is_claude_dead = False

                # СТОРОЖЕВОЙ ТАЙМЕР НЕПОДВИЖНОЙ ЦЕЛИ (ANTI-STUCK / ANTI-CORPSE WATCHDOG)
                # Если цель неподвижна на одном месте более 2.5 сек под атакой -> это труп/текстура!
                if best_enemy:
                    if stationary_target_pos is None or np.hypot(best_enemy[0] - stationary_target_pos[0], best_enemy[1] - stationary_target_pos[1]) > 20:
                        stationary_target_pos = best_enemy
                        stationary_target_time = now
                    else:
                        if (now - stationary_target_time) > 2.5:
                            corpse_zones.append({"pos": best_enemy, "expiry": now + 20.0})
                            print(f"\n[🛡️ ANTI-CORPSE] Цель {best_enemy} неподвижна >2.5с (труп/текстура). В черный список на 20с!")
                            best_enemy = None
                            tracked_enemy_pos = None
                            stationary_target_pos = None
                else:
                    stationary_target_pos = None

                # Вычисляем крипа для фарма (ТОЛЬКО если вражеских героев нет рядом!)
                best_farm = None
                dist_farm = 9999.0
                if farm_targets and not best_enemy and (now - last_known_enemy_time > 1.2):
                    farm_targets.sort(key=lambda t: (t[0] - ref_pt[0])**2 + (t[1] - ref_pt[1])**2)
                    tracked_farm_pos = (farm_targets[0][0], farm_targets[0][1])
                    tracked_farm_time = now
                    best_farm = tracked_farm_pos
                    dist_farm = np.sqrt((best_farm[0] - ref_pt[0])**2 + (best_farm[1] - ref_pt[1])**2)
                elif tracked_farm_pos and not best_enemy and (now - tracked_farm_time) < 0.60:
                    best_farm = tracked_farm_pos
                    dist_farm = np.sqrt((best_farm[0] - ref_pt[0])**2 + (best_farm[1] - ref_pt[1])**2)
                else:
                    best_farm = None
                    tracked_farm_pos = None

                # Надежная проверка успешного фарма крипа (без ложных срабатываний):
                if best_farm:
                    last_farm_pos = best_farm
                    if farm_attack_start_time == 0.0:
                        farm_attack_start_time = now
                    farm_disappeared_since = 0.0
                elif last_farm_pos and not best_enemy:
                    if farm_disappeared_since == 0.0:
                        farm_disappeared_since = now
                    elif (now - farm_disappeared_since) >= 2.0:
                        # Крип реально уничтожен, если мы непрерывно атаковали его не менее 2.5 секунд
                        if (now - farm_attack_start_time) >= 2.5:
                            reward_sys.add_reward(10, "MINION / CREEP FARMED")
                            rl_brain.reward_farm(25.0, "CREEP SLAIN (LAST HIT)")
                            rl_brain.reward_roam(15.0, "FOUND JUNGLE/LANE FARM")
                        last_farm_pos = None
                        farm_attack_start_time = 0.0
                        farm_disappeared_since = 0.0

                # Сторожевой таймер защиты от союзных крипов (Ally Creep Watchdog):
                # Если бьем крипа дольше 4.5 секунд, а он не умирает -> это наш союзный крип! В черный список!
                if best_farm and farm_attack_start_time > 0.0:
                    if (now - farm_attack_start_time) > 4.5:
                        print(f"\n[🛡️ ALLY-CREEP WATCHDOG] Цель {best_farm} неуязвима >4.5с (наш союзный крип!). В черный список на 20с!")
                        corpse_zones.append({"pos": best_farm, "expiry": now + 20.0})
                        best_farm = None
                        tracked_farm_pos = None
                        last_farm_pos = None
                        farm_attack_start_time = 0.0
                        last_combat_or_farm_time = 0.0  # Принудительный толчок бежать вперед!

                # Настроение ИИ для HUD
                ai_mood_text, ai_mood_color = rl_brain.get_ai_mood(
                    hp_ratio=hp_ratio,
                    enemy_present=(best_enemy is not None),
                    can_combo=can_full_combo,
                    is_dead=is_claude_dead,
                    has_farm=(best_farm is not None)
                )

                # Отрисовка активных объектов на холсте
                if self_target:
                    sx, sy = int(self_target[0]), int(self_target[1])
                    cv2.rectangle(annotated_frame, (sx - 35, sy - 12), (sx + 35, sy + 12), (0, 255, 0), 2)
                    hp_text = f"Claude ({int(hp_ratio * 100)}% HP)"
                    cv2.putText(annotated_frame, hp_text, (sx - 45, sy - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                for cz in corpse_zones:
                    cx, cy = int(cz["pos"][0]), int(cz["pos"][1])
                    cv2.circle(annotated_frame, (cx, cy), 35, (90, 90, 90), 1)
                    cv2.putText(annotated_frame, "DEAD CORPSE (IGNORED)", (cx - 70, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)

                for ex, ey, ew, eh in enemy_heroes:
                    bx1 = int(ex - ew / 2)
                    by1 = int(ey - eh / 2)
                    bx2 = int(ex + ew / 2)
                    by2 = int(ey + eh / 2)
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, "hp_enemy", (bx1, max(15, by1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                for fx, fy, fw, fh in farm_targets:
                    bx1 = int(fx - fw / 2)
                    by1 = int(fy - fh / 2)
                    bx2 = int(fx + fw / 2)
                    by2 = int(fy + fh / 2)
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                    cv2.putText(annotated_frame, "creep", (bx1, max(15, by1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                for bx, by, bw, bh in detected_bushes:
                    x1 = int(bx - bw / 2)
                    y1 = int(by - bh / 2)
                    x2 = int(bx + bw / 2)
                    y2 = int(by + bh / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (34, 139, 34), 1)
                    cv2.putText(annotated_frame, "bush", (x1, max(15, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (34, 139, 34), 1)

                ai_state = "IDLE / PATROL"
                ai_color = (200, 200, 200)

                # ============================================================
                # ТАКТИЧЕСКИЙ МОЗГ: ПРИНЯТИЕ РЕШЕНИЙ
                # ============================================================
                if is_claude_dead:
                    joystick.release()
                    ai_state = "DEAD (RESPAWNING)"
                    ai_color = (0, 0, 255)

                elif claude_macro:
                    # РЕЖИМ ПАНИКИ: ЗДОРОВЬЕ КЛОДА КРИТИЧЕСКИ МАЛО (< 32%)
                    if hp_ratio < 0.32:
                        combat_stance = "PANIC_RETREAT"
                        ai_state = "PANIC RETREAT (HP CRITICAL!)"
                        ai_color = (255, 0, 220)
                        if s2_ok and (now - last_s2_time) >= S2_COOLDOWN_SEC:
                            claude_macro.panic_retreat_s2()
                            last_s2_time = now
                        # Бег назад к вышке БЕЗ отрыва пальца
                        safe_point = (ref_pt[0] - 180, ref_pt[1] + 90)
                        joystick.move_towards(ref_pt, safe_point)
                        last_joystick_time = now
                        if (now - last_regen_time) >= REGEN_COOLDOWN_SEC:
                            claude_macro.trigger_regen()
                            last_regen_time = now

                    # СИТУАЦИЯ 0: СПРИНТ НА ЛИНИЮ СО СТАРТА МАТЧА ИЛИ ПОСЛЕ ВОЗРОЖДЕНИЯ (RUSH TO LANE)
                    elif now < rush_to_lane_end_time and not (best_enemy and dist_enemy < 350.0):
                        ai_state = f"RUSH TO LANE ({int(max(0, rush_to_lane_end_time - now))}s)"
                        ai_color = (0, 255, 255)
                        # Прямой скоростной марш по вектору линии вперед из базы
                        lane_target = (ref_pt[0] + 300, ref_pt[1] - 80)
                        joystick.move_towards(ref_pt, lane_target)
                        last_joystick_time = now
                        last_combat_or_farm_time = now

                        # Стартовая авто-покупка предмета и прокачка 1-го скилла раз в 3 секунды
                        if (now - last_quick_buy_time) >= 3.0:
                            claude_macro.quick_buy_item()
                            claude_macro.level_up_skills()
                            last_quick_buy_time = now

                    # СИТУАЦИЯ 1: ВРАЖЕСКИЙ ГЕРОЙ В ПОЛЕ ЗРЕНИЯ (Q-LEARNING COMBAT ENGINE)
                    elif best_enemy:
                        last_attacked_enemy_pos = best_enemy
                        last_attacked_enemy_time = now
                        last_combat_or_farm_time = now
                        current_roam_target = None

                        # Награда за обнаружение врага
                        if not reward_sys.enemy_was_visible or (now - reward_sys.last_scout_reward_time > 8.0):
                            reward_sys.add_reward(15, "ENEMY SPOTTED IN SCOUT!")
                            reward_sys.last_scout_reward_time = now
                            rl_brain.reward_roam(30.0, "FOUND ENEMY HERO")
                        reward_sys.enemy_was_visible = True

                        # Обучающийся тактический выбор через Q-Learning
                        combat_act = rl_brain.evaluate_combat_stance(
                            dist_enemy=dist_enemy,
                            can_combo=can_full_combo,
                            is_turret_near=turret_danger,
                            hp_self_ratio=hp_ratio
                        )

                        # 1. ЗАЩИТА: Опасность вражеской вышки -> отход
                        if combat_act == "TURRET_RETREAT":
                            ai_state = "TURRET DANGER (RETREAT)"
                            ai_color = (0, 0, 255)
                            rl_brain.reward_combat(-40.0, "WALKED INTO TURRET")
                            if enemy_turrets:
                                joystick.kite_away(ref_pt, (enemy_turrets[0][0], enemy_turrets[0][1]))
                            else:
                                safe_point = (ref_pt[0] - 180, ref_pt[1] + 90)
                                joystick.move_towards(ref_pt, safe_point)
                            last_joystick_time = now

                        # 2. СПАСЕНИЕ ЖИЗНИ: Здоровье Клода просело -> тактический отход под вышку
                        elif combat_act == "TACTICAL_RETREAT":
                            ai_state = "TACTICAL RETREAT (LOW HP)"
                            ai_color = (255, 0, 200)
                            safe_point = (ref_pt[0] - 180, ref_pt[1] + 90)
                            joystick.move_towards(ref_pt, safe_point)
                            last_joystick_time = now
                            if s2_ok and (now - last_s2_time) >= S2_COOLDOWN_SEC:
                                claude_macro.panic_retreat_s2()
                                last_s2_time = now
                            if (now - last_regen_time) >= REGEN_COOLDOWN_SEC:
                                claude_macro.trigger_regen()
                                last_regen_time = now

                        # 3. КАЙТИНГ И ПОУК: Враг напирает -> шаг назад + С1 (кража скорости) + тычки
                        elif combat_act == "KITE_AND_POKE":
                            ai_state = f"Q-KITING ({int(dist_enemy)}px)"
                            ai_color = (0, 165, 255)
                            joystick.kite_away(ref_pt, best_enemy)
                            last_joystick_time = now
                            if s1_ok and (now - last_s1_time) >= S1_COOLDOWN_SEC:
                                claude_macro.fire_skill_1_fast()
                                last_s1_time = now
                            if (now - last_attack_time) >= BASIC_ATTACK_COOLDOWN_SEC:
                                claude_macro.fire_basic_attack_fast()
                                last_attack_time = now

                        # 4. РАССТРЕЛ ИЗ SWEET-SPOT: Непрерывный орбитальный стрейф вокруг врага на ходу!
                        elif combat_act == "SWEET_SPOT_BURST":
                            ai_state = f"Q-BURST STRAFE ({int(dist_enemy)}px)"
                            ai_color = (0, 255, 0)
                            # Орбитальный кайт: стрейф по касательной (никогда не стоим на месте!)
                            dx = best_enemy[0] - ref_pt[0]
                            dy = best_enemy[1] - ref_pt[1]
                            strafe_target = (ref_pt[0] - dy * 0.45, ref_pt[1] + dx * 0.45)
                            joystick.move_towards(ref_pt, strafe_target)
                            last_joystick_time = now
                            if s1_ok and (now - last_s1_time) >= S1_COOLDOWN_SEC:
                                claude_macro.fire_skill_1_fast()
                                last_s1_time = now
                            if (now - last_attack_time) >= BASIC_ATTACK_COOLDOWN_SEC:
                                claude_macro.fire_basic_attack_fast()
                                last_attack_time = now

                        # 5. РЕШАЮЩИЙ ВРЫВ: Добивание фулл-комбо Клода
                        elif combat_act == "DIVE_ALL_IN":
                            if can_full_combo and (now - last_combo_time) >= ULT_COOLDOWN_SEC and hp_ratio > 0.60:
                                ai_state = "DIVE ALL-IN (BURST ULT!)"
                                ai_color = (255, 0, 255)
                                claude_macro.trigger_engage_ultimate()
                                last_combo_time = now
                                last_s2_time = now
                                last_s1_time = now
                                reward_sys.add_reward(30, "CLAUDE ALL-IN COMBO!")
                                rl_brain.reward_combat(35.0, "BURST COMBO EXECUTED")
                                rl_brain.record_best_moment(
                                    moment_type="BURST_COMBO",
                                    reward=45.0,
                                    description="Фулл-прокаст комбо S2 -> Врыв -> Ult -> S1",
                                    hp_left_pct=int(hp_ratio * 100)
                                )
                                reward_sys.highlights_count = len(rl_brain.best_moments)
                            else:
                                ai_state = f"CHASING ENEMY ({int(dist_enemy)}px)"
                                ai_color = (255, 255, 0)
                                joystick.move_towards(ref_pt, best_enemy)
                                last_joystick_time = now
                                if (now - last_attack_time) >= BASIC_ATTACK_COOLDOWN_SEC:
                                    claude_macro.fire_basic_attack_fast()
                                    last_attack_time = now

                    # СИТУАЦИЯ 2: ГЕРОЕВ НЕТ, НО ЕСТЬ КРИПЫ/МОНСТРЫ -> Q-LEARNING АВТО-ФАРМ
                    elif best_farm:
                        reward_sys.enemy_was_visible = False
                        combat_stance = "FARMING"
                        last_farm_pos = best_farm
                        last_combat_or_farm_time = now
                        current_roam_target = None

                        # Принятие тактического решения фарма через Q-Learning
                        farm_act = rl_brain.evaluate_farm_stance(
                            dist_farm=dist_farm,
                            s1_ok=s1_ok,
                            hp_self_ratio=hp_ratio
                        )

                        if farm_act == "FARM_APPROACH":
                            ai_state = f"Q-FARM APPROACH ({int(dist_farm)}px)"
                            ai_color = (0, 255, 255)
                            joystick.move_towards(ref_pt, best_farm)
                            last_joystick_time = now
                            if (now - last_attack_time) >= BASIC_ATTACK_COOLDOWN_SEC:
                                claude_macro.fire_basic_attack_fast()
                                last_attack_time = now

                        elif farm_act == "FARM_KITE_BACK":
                            ai_state = f"Q-FARM KITING ({int(dist_farm)}px)"
                            ai_color = (0, 200, 255)
                            joystick.kite_away(ref_pt, best_farm)
                            last_joystick_time = now
                            if (now - last_attack_time) >= BASIC_ATTACK_COOLDOWN_SEC:
                                claude_macro.fire_basic_attack_fast()
                                last_attack_time = now

                        elif farm_act == "FARM_S1_AOE":
                            ai_state = f"Q-FARM S1 AOE ({int(dist_farm)}px)"
                            ai_color = (0, 255, 120)
                            # С1 на фарме отдается ТОЛЬКО если перед нами пачка (>=2 крипа) и прошел кулдаун
                            if s1_ok and len(farm_targets) >= 2 and (now - last_s1_time) >= 8.0:
                                claude_macro.fire_skill_1_fast()
                                last_s1_time = now
                                rl_brain.reward_farm(20.0, "S1 AOE CAST ON WAVE")
                            if (now - last_attack_time) >= BASIC_ATTACK_COOLDOWN_SEC:
                                claude_macro.fire_basic_attack_fast()
                                last_attack_time = now

                        elif farm_act == "FARM_SWEET_SPOT":
                            ai_state = f"Q-FARM ORB-WALK ({int(dist_farm)}px)"
                            ai_color = (0, 255, 255)
                            # Орб-вокинг: непрерывное движение вокруг крипа во время выстрелов
                            dx = best_farm[0] - ref_pt[0]
                            dy = best_farm[1] - ref_pt[1]
                            strafe_farm = (ref_pt[0] - dy * 0.40, ref_pt[1] + dx * 0.40)
                            joystick.move_towards(ref_pt, strafe_farm)
                            last_joystick_time = now
                            if (now - last_attack_time) >= BASIC_ATTACK_COOLDOWN_SEC:
                                claude_macro.fire_basic_attack_fast()
                                last_attack_time = now

                        # Тактический авто-хил во время фарма (только при низком HP и по кулдауну 60с)
                        if hp_ratio < 0.50 and (now - last_regen_time) >= REGEN_COOLDOWN_SEC:
                            claude_macro.trigger_regen()
                            last_regen_time = now

                    # СИТУАЦИЯ 3: НИКОГО НЕТ -> ПУШ ЛИНИИ / АВТОНОМНАЯ РАЗВЕДКА
                    else:
                        reward_sys.enemy_was_visible = False
                        combat_stance = "SCOUTING"

                        # 1. Если на экране есть союзные миньоны, марширующие вперед - пушим линию за ними!
                        if ally_minions:
                            ally_minions.sort(key=lambda m: m[0], reverse=True)  # Самый передовой миньон
                            lead_m = ally_minions[0]
                            push_target = (lead_m[0] - 70, lead_m[1])
                            ai_state = f"LANE PUSH (WAVE: {len(ally_minions)})"
                            ai_color = (0, 255, 128)
                            joystick.move_towards(ref_pt, push_target)
                            last_joystick_time = now
                            last_combat_or_farm_time = now
                        else:
                            # 2. Мягкий штраф за долгое бездействие (только после 25 сек пути)
                            if (now - last_combat_or_farm_time) >= IDLE_PENALTY_INTERVAL:
                                reward_sys.penalize_idle(3, "IDLE (EXPLORING NEW SECTOR)")
                                rl_brain.reward_roam(-3.0, "EMPTY PATH (IDLE)")
                                current_roam_target = None
                                last_combat_or_farm_time = now

                            # 3. Выбор направления: бежит уверенно в сторону выбранного сектора
                            if (now - roam_leg_start_time) >= ROAM_LEG_DURATION or current_roam_target is None:
                                current_roam_action, current_roam_target = rl_brain.choose_roam_direction(ref_pt, now - last_known_enemy_time)
                                roam_leg_start_time = now

                            ai_state = f"Q-ROAM: {rl_brain.get_current_roam_info()}"
                            ai_color = (0, 200, 255)

                            # 4. Непрерывный бег по выбранному Q-маршруту БЕЗ отрыва пальца
                            if (now - last_joystick_time) >= 0.08 and current_roam_target:
                                joystick.move_towards(ref_pt, current_roam_target)
                                last_joystick_time = now

                # 6. Отрисовка HUD на кадре
                c_s1 = (0, 255, 0) if s1_ok else (0, 0, 255)
                c_s2 = (0, 255, 0) if s2_ok else (0, 0, 255)
                c_ult = (0, 255, 0) if ult_ok else (0, 0, 255)

                cd_s1 = max(0.0, S1_COOLDOWN_SEC - (now - last_s1_time))
                cd_s2 = max(0.0, S2_COOLDOWN_SEC - (now - last_s2_time))
                cd_ult = max(0.0, ULT_COOLDOWN_SEC - (now - last_combo_time))

                t_s1 = "S1" if s1_ok else f"S1:{cd_s1:.0f}s"
                t_s2 = "S2" if s2_ok else f"S2:{cd_s2:.0f}s"
                t_ult = "ULT" if ult_ok else f"ULT:{cd_ult:.0f}s"

                cv2.putText(annotated_frame, t_s1, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_s1, 2)
                cv2.putText(annotated_frame, t_s2, (85, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_s2, 2)
                cv2.putText(annotated_frame, t_ult, (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_ult, 2)

                # Вывод текущего состояния интеллекта
                cv2.putText(annotated_frame, f"AI: {ai_state}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ai_color, 2)

                # Линия прицела и маркер цели
                if best_enemy:
                    tx, ty = int(best_enemy[0]), int(best_enemy[1])
                    cv2.circle(annotated_frame, (tx, ty), 18, (0, 0, 255), 2)
                    sx, sy = int(ref_pt[0]), int(ref_pt[1])
                    cv2.line(annotated_frame, (sx, sy), (tx, ty), (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"{int(dist_enemy)}px", ((sx + tx)//2, (sy + ty)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                elif best_farm:
                    fx, fy = int(best_farm[0]), int(best_farm[1])
                    cv2.circle(annotated_frame, (fx, fy), 18, (0, 255, 255), 2)
                    sx, sy = int(ref_pt[0]), int(ref_pt[1])
                    cv2.line(annotated_frame, (sx, sy), (fx, fy), (0, 200, 255), 2)

                # Выбор актуального Q-статуса для HUD
                if best_enemy:
                    current_q_info = rl_brain.get_current_combat_info()
                elif best_farm:
                    current_q_info = rl_brain.get_current_farm_info()
                else:
                    current_q_info = rl_brain.get_current_roam_info()

                # Отрисовка плашки морковок, настроения и Q-Brain
                reward_sys.render_hud(annotated_frame, ai_mood_text, ai_mood_color, current_q_info)

                # 7. Отображение окна предпросмотра
                fps = 1.0 / max(0.001, (time.time() - start_time))

                if SHOW_PREVIEW_WINDOW:
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f} | CLAUDE Q-LEARNING AI", (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "[Q] Exit  [Space/F] Combo", (frame_w - 270, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                    cv2.imshow(cv_window_name, annotated_frame)

                    # Аппаратное исключение окна из захвата DWM (устранение зеркального коридора)
                    if not window_affinity_set:
                        cv_hwnd = win32gui.FindWindow(None, cv_window_name)
                        if cv_hwnd:
                            ctypes.windll.user32.SetWindowDisplayAffinity(cv_hwnd, 0x11)
                            init_x = 20
                            init_y = max(20, screen_height - PREVIEW_HEIGHT - 60)
                            win32gui.SetWindowPos(cv_hwnd, win32con.HWND_TOPMOST, init_x, init_y, PREVIEW_WIDTH, PREVIEW_HEIGHT, win32con.SWP_SHOWWINDOW)
                            window_affinity_set = True

                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:
                        break
                    elif key in [ord(' '), ord('f'), ord('F')]:
                        if claude_macro:
                            print("\n[MANUAL] Принудительный запуск комбо!")
                            claude_macro.trigger_engage_ultimate()
                else:
                    time.sleep(0.001)

                frame_count += 1
                if frame_count % 20 == 0:
                    status_line = f"FPS: {fps:.1f} | 🥕 {reward_sys.carrots} | K/D: {reward_sys.kills}/{reward_sys.deaths} | {ai_state}"
                    print(f"\r[*] {status_line}                     ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[*] Остановка.")
    finally:
        joystick.close()
        rl_brain.save_memory()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
