"""
Централизованный конфигурационный слой проекта (Config Layer V2)
================================================================
Единый источник истины для параметров окружения, нейросети,
координат кнопок, таймеров и боевых дистанций.
"""

import os

# ==============================================================================
# 01.1 Screen Configuration
# ==============================================================================
SCREEN_WIDTH = 1544
SCREEN_HEIGHT = 720

PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 270

SHOW_PREVIEW_WINDOW = True

# ==============================================================================
# 01.2 Device / ADB Configuration
# ==============================================================================
DEVICE_SERIAL = os.getenv("MLBB_DEVICE_SERIAL", "R3CT90BBMTX")
ADB_BINARY = os.getenv("MLBB_ADB_BINARY", "adb")

# ==============================================================================
# 01.3 YOLO Configuration
# ==============================================================================
WEIGHTS_PATH = os.getenv(
    "MLBB_WEIGHTS_PATH",
    r"models\yolo\v4\best.pt" if os.path.exists(r"models\yolo\v4\best.pt")
    else r"runs\detect\mlbb_training\v4_run\weights\best.pt"
)

YOLO_DEVICE = "0"
YOLO_IMGSZ = 480

CONF_HERO_THRESHOLD = 0.40
CONF_MINION_THRESHOLD = 0.40
CONF_BUFF_THRESHOLD = 0.22
CONF_TURRET_THRESHOLD = 0.25

# ==============================================================================
# 01.4 Tactical Distance Configuration
# ==============================================================================
KITING_DANGER_DIST_PX = 280.0
SWEET_SPOT_MAX_DIST_PX = 430.0
TURRET_DANGER_DIST_PX = 420.0

# ==============================================================================
# 01.5 Cooldowns & Intervals (Seconds)
# ==============================================================================
BASIC_ATTACK_COOLDOWN_SEC = 0.20
COMBO_COOLDOWN_SEC = 10.0

S1_COOLDOWN_SEC = 5.5
S2_COOLDOWN_SEC = 10.0
ULT_COOLDOWN_SEC = 40.0

LEVELUP_INTERVAL_SEC = 25.0
REGEN_COOLDOWN_SEC = 60.0

JOYSTICK_STEP_SEC = 0.28
ROAM_LEG_DURATION = 3.0
IDLE_PENALTY_INTERVAL = 25.0

# ==============================================================================
# 01.6 Touch Coordinates (Exact 1:1 match with V1 calibration)
# ==============================================================================
# Виртуальный джойстик
JOYSTICK_CENTER_X = 288
JOYSTICK_CENTER_Y = 560
JOYSTICK_RADIUS = 110

# Кнопки способностей и атаки (сырые координаты оцифровщика / калибровка Клода)
COORD_SKILL_1 = (180, 1088)       # Искусство воровства (1088, 540)
COORD_SKILL_2 = (328, 1228)       # Зеркальное отражение (1228, 392)
COORD_ULTIMATE = (345, 1051)      # Буйство стрельбы (1051, 375)
COORD_FLICKER = (93, 850)         # Боевое заклинание: Вспышка (850, 627)
COORD_BASIC_ATTACK = (160, 1350)  # Кнопка атаки (1350, 560)

SWAP_XY = True

# Быстрые кнопки на экране
PLUS_BUTTONS = [(1051, 305), (1088, 465), (1228, 320)]
QUICK_BUY_COORDS = (1220, 160)
REGEN_BUTTON_COORDS = (815, 640)

# Координаты кнопок способностей для оптического сенсора SkillStateChecker
SKILL_CHECK_COORDS = {
    "s1": (1088, 540),
    "s2": (1228, 392),
    "ult": (1051, 375)
}

# ==============================================================================
# 01.7 Humanizer Configuration
# ==============================================================================
SPATIAL_JITTER_SIGMA = 3.0
SPATIAL_MAX_OFFSET = 6

HOLD_DURATION_MEAN_MS = 65.0
HOLD_DURATION_MIN_MS = 45.0
HOLD_DURATION_MAX_MS = 95.0
