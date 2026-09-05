"""
Автономный самообучающийся мозг Клода (Claude Q-Learning Brain)
========================================================================
Реализует алгоритм обучения с подкреплением (Reinforcement Learning / Q-Learning):
  1. Адаптивная разведка (Autonomous Exploration):
     Бот выбирает направления исследования карты (линия, река, лес, вышка)
     на основе матрицы Q-значений, а не жестко зашитых координат.
  2. Обучение на ошибках (Trial & Error):
     - Нашел врага / крипа -> положительное подкрепление (+морковки).
     - Полез под вышку / умер -> жесткое наказание (-50% морковок) и падение Q-веса.
  3. Долгосрочная память (Long-term Memory):
     Матрица опыта сохраняется в файл `q_brain.json` и накапливается от игры к игре.
"""

import os
import sys
import json
import time
import random
from typing import Tuple, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


class ClaudeRLBrain:
    # --------------------------------------------------------------------------
    # Направления исследования карты (относительные векторы смещения от Клода)
    # --------------------------------------------------------------------------
    ROAM_ACTIONS = {
        "LANE_ADVANCE":     {"name": "Марш по линии (прямо)",    "dx": 230,  "dy": -60},
        "LANE_PUSH":        {"name": "Пуш линии (вперед)",       "dx": 190,  "dy": -20},
        "RIVER_SCOUT":      {"name": "Контроль реки (центр)",    "dx": 130,  "dy": 90},
        "FLANK_ADVANCE":    {"name": "Обход по флангу (верх)",   "dx": 110,  "dy": -130},
        "CREEP_INTERCEPT":  {"name": "Перехват крипов (низ)",    "dx": 160,  "dy": 40},
    }

    # --------------------------------------------------------------------------
    # Боевые тактические действия
    # --------------------------------------------------------------------------
    DUEL_ACTIONS = ["KITE_AND_POKE", "SWEET_SPOT_BURST", "DIVE_ALL_IN", "TACTICAL_RETREAT"]
    COMBAT_ACTIONS = DUEL_ACTIONS + ["TURRET_RETREAT"]

    # --------------------------------------------------------------------------
    # Тактические действия фарма крипов
    # --------------------------------------------------------------------------
    FARM_ACTIONS = ["FARM_APPROACH", "FARM_KITE_BACK", "FARM_SWEET_SPOT", "FARM_S1_AOE"]

    def __init__(
        self,
        memory_file: str = "q_brain.json",
        learning_rate: float = 0.25,
        discount_factor: float = 0.85,
        exploration_rate: float = 0.20,
    ):
        self.memory_file = memory_file
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

        # Q-таблицы (состояние -> {действие: вес})
        self.roam_q_table: Dict[str, Dict[str, float]] = {}
        self.farm_q_table: Dict[str, Dict[str, float]] = {}
        self.combat_q_table: Dict[str, Dict[str, float]] = {}

        # Банк лучших моментов («Золотой фонд» памяти Клода)
        self.best_moments: List[Dict] = []
        self.recent_action_history: List[Dict] = []  # Скользящее окно последних действий
        self.last_highlight_text: str = ""

        # Текущие решения для обучения
        self.last_roam_state: Optional[str] = None
        self.last_roam_action: Optional[str] = None
        self.last_roam_time: float = 0.0

        self.last_farm_state: Optional[str] = None
        self.last_farm_action: Optional[str] = None

        self.last_combat_state: Optional[str] = None
        self.last_combat_action: Optional[str] = None

        self.total_decisions: int = 0
        self.successful_learnings: int = 0

        self._load_memory()

    def _load_memory(self) -> None:
        """Загружает накопленный опыт и лучшие моменты из файла q_brain.json."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.roam_q_table = data.get("roam_q", {})
                    self.farm_q_table = data.get("farm_q", {})
                    self.combat_q_table = data.get("combat_q", {})
                    self.best_moments = data.get("best_moments", [])
                    self.total_decisions = data.get("total_decisions", 0)
                learned_count = len(self.roam_q_table) + len(self.farm_q_table) + len(self.combat_q_table)
                print(f"[🧠 Q-BRAIN] Загружена память ИИ: {learned_count} состояний, {len(self.best_moments)} лучших моментов из {self.memory_file}")
            except Exception as e:
                print(f"[!] Ошибка загрузки памяти Q-Brain: {e}")

    def save_memory(self) -> None:
        """Сохраняет опыт и лучшие моменты на диск для следующих матчей."""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump({
                    "roam_q": self.roam_q_table,
                    "farm_q": self.farm_q_table,
                    "combat_q": self.combat_q_table,
                    "best_moments": self.best_moments[-50:],  # Хранить топ-50 лучших моментов
                    "total_decisions": self.total_decisions,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[!] Ошибка сохранения памяти Q-Brain: {e}")

    # ==========================================================================
    # 1. АВТОНОМНАЯ РАЗВЕДКА (ОБУЧЕНИЕ МАРШРУТАМ И НАПРАВЛЕНИЯМ)
    # ==========================================================================
    def choose_roam_direction(self, ref_pt: Tuple[float, float], time_since_enemy: float) -> Tuple[str, Tuple[int, int]]:
        """
        Выбирает следующее направление движения на основе Q-обучения (Epsilon-Greedy).
        Бот исследует карту, а не ходит по зашитой траектории!
        """
        # Дискретизация состояния разведки: сколько времени прошло с последнего контакта
        if time_since_enemy < 5.0:
            state = "SCOUT_HOT_ZONE"      # Недавно был бой -> проверить фланги
        elif time_since_enemy < 15.0:
            state = "SCOUT_MID_SEARCH"    # Ищем крипов и линии
        else:
            state = "SCOUT_DEEP_PATROL"   # Долгий поиск -> патруль реки и вышек

        self.last_roam_state = state

        # Инициализация Q-строки состояния
        if state not in self.roam_q_table:
            self.roam_q_table[state] = {act: 0.0 for act in self.ROAM_ACTIONS.keys()}

        actions = list(self.ROAM_ACTIONS.keys())

        # Epsilon-Greedy: иногда пробуем новое направление (разведка),
        # но чаще выбираем самое прибыльное по опыту (эксплуатация)
        if random.random() < self.epsilon:
            chosen_action = random.choice(actions)
        else:
            # Выбираем действие с максимальным Q-значением
            q_vals = self.roam_q_table[state]
            max_q = max(q_vals.values())
            best_acts = [act for act, q in q_vals.items() if q == max_q]
            chosen_action = random.choice(best_acts)

        self.last_roam_action = chosen_action
        self.last_roam_time = time.time()
        self.total_decisions += 1
        self.log_action("ROAM", state, chosen_action)

        action_data = self.ROAM_ACTIONS[chosen_action]
        target_pt = (int(ref_pt[0] + action_data["dx"]), int(ref_pt[1] + action_data["dy"]))
        return chosen_action, target_pt

    def reward_roam(self, reward_amount: float, reason: str = "") -> None:
        """Поощряет выбранное направление разведки (например, нашел врага/крипа)."""
        if not self.last_roam_state or not self.last_roam_action:
            return

        s = self.last_roam_state
        a = self.last_roam_action

        old_q = self.roam_q_table[s].get(a, 0.0)
        # Формула Беллмана для Q-learning: Q(s,a) = Q(s,a) + alpha * [R - Q(s,a)]
        new_q = old_q + self.alpha * (reward_amount - old_q)
        self.roam_q_table[s][a] = round(new_q, 2)
        self.successful_learnings += 1

        if abs(reward_amount) >= 15:
            print(f"[🧠 ОБУЧЕНИЕ МАРШРУТА] Действие '{a}' в состоянии '{s}' получило Q={new_q:.1f} ({reason})")
            self.save_memory()

    def penalize_wall_stuck(self, reason: str = "WALL COLLISION") -> Tuple[int, int]:
        """
        Штрафует направление за утыкание в препятствие (стену) и возвращает вектор отскока.
        """
        if self.last_roam_state and self.last_roam_action:
            s = self.last_roam_state
            a = self.last_roam_action
            old_q = self.roam_q_table[s].get(a, 0.0)
            new_q = old_q - 25.0
            self.roam_q_table[s][a] = round(new_q, 2)
            print(f"\n[🧱 СТЕНА / ПРЕПЯТСТВИЕ] Вектор '{a}' застрял! Q снижен до {new_q:.1f} ({reason})")
            self.save_memory()
        # Вектор обхода препятствия (скольжение вперед под углом)
        return (140, 90)

    def get_current_roam_info(self) -> str:
        """Возвращает читаемый статус текущего решения разведки с Q-значением."""
        if not self.last_roam_state or not self.last_roam_action:
            return "IDLE_SEARCH"
        q_val = self.roam_q_table.get(self.last_roam_state, {}).get(self.last_roam_action, 0.0)
        action_name = self.ROAM_ACTIONS.get(self.last_roam_action, {}).get("name", self.last_roam_action)
        return f"{self.last_roam_action} [Q={q_val:.1f}]"

    def get_ai_mood(
        self,
        hp_ratio: float,
        enemy_present: bool,
        can_combo: bool,
        is_dead: bool,
        has_farm: bool
    ) -> Tuple[str, Tuple[int, int, int]]:
        """
        Определяет эмоциональное/тактическое настроение ИИ для HUD дашборда:
        (Mood_Text, BGR_Color)
        """
        if is_dead:
            return "DEAD (RESPAWNING)", (0, 0, 255)
        if hp_ratio < 0.32:
            return "PANIC RETREAT (HP CRITICAL!)", (255, 0, 220)
        if enemy_present:
            if hp_ratio > 0.65 and can_combo:
                return "PREDATOR (BURST ALL-IN!)", (0, 0, 255)
            return "TACTICAL KITING (COMBAT)", (0, 165, 255)
        if has_farm:
            return "CHILL FARMING (MINIONS)", (0, 255, 255)
        return f"Q-ROAM ({self.last_roam_action or 'SCOUT'})", (0, 255, 0)

    # ==========================================================================
    # 2. ОБУЧЕНИЕ ФАРМУ КРИПОВ (CREEP FARMING Q-LEARNING)
    # ==========================================================================
    def evaluate_farm_stance(
        self,
        dist_farm: float,
        s1_ok: bool,
        hp_self_ratio: float = 1.0,
    ) -> str:
        """
        Q-Learning выбор действий при фарме крипов:
        - Обучается держать безопасную дистанцию (260-400px), чтобы крипы ближнего боя не наносили урон.
        - Обучается отдавать Скилл 1 по крипам для разгона стаков скорости атаки и АОЕ зачистки.
        - Добивает крипов авто-атаками из идеальной позиции.
        """
        if dist_farm > 420.0:
            dist_bucket = "CREEP_FAR"
        elif dist_farm < 260.0:
            dist_bucket = "CREEP_DANGER_CLOSE"
        else:
            dist_bucket = "CREEP_SWEET_SPOT"

        hp_cat = "SAFE" if hp_self_ratio >= 0.50 else "LOW"
        state = f"{dist_bucket}_S1_{s1_ok}_{hp_cat}"
        self.last_farm_state = state

        if state not in self.farm_q_table:
            self.farm_q_table[state] = {
                "FARM_APPROACH": 30.0 if dist_bucket == "CREEP_FAR" else -10.0,
                "FARM_KITE_BACK": 35.0 if dist_bucket == "CREEP_DANGER_CLOSE" else -5.0,
                "FARM_SWEET_SPOT": 40.0 if dist_bucket == "CREEP_SWEET_SPOT" else 5.0,
                "FARM_S1_AOE": 45.0 if (dist_bucket == "CREEP_SWEET_SPOT" and s1_ok) else 10.0,
            }

        q_vals = self.farm_q_table[state]
        if random.random() < 0.10:  # 10% исследование вариаций фарма
            chosen = random.choice(self.FARM_ACTIONS)
        else:
            max_q = max(q_vals.values())
            best_acts = [act for act, q in q_vals.items() if q == max_q]
            chosen = random.choice(best_acts)

        self.last_farm_action = chosen
        self.log_action("FARM", state, chosen)
        return chosen

    def reward_farm(self, reward_amount: float, reason: str = "") -> None:
        """Поощряет или наказывает тактику фарма (убийство крипа, АОЕ по пачке, урон)."""
        if not self.last_farm_state or not self.last_farm_action:
            return

        s = self.last_farm_state
        a = self.last_farm_action

        old_q = self.farm_q_table[s].get(a, 0.0)
        new_q = old_q + self.alpha * (reward_amount - old_q)
        self.farm_q_table[s][a] = round(new_q, 2)

        if abs(reward_amount) >= 15:
            print(f"[🌾 ОБУЧЕНИЕ КРИПОВ] Тактика '{a}' в '{s}' обновлена до Q={new_q:.1f} ({reason})")
            self.save_memory()

    def get_current_farm_info(self) -> str:
        """Статус текущего решения фарма для HUD."""
        if not self.last_farm_state or not self.last_farm_action:
            return "FARMING"
        q_val = self.farm_q_table.get(self.last_farm_state, {}).get(self.last_farm_action, 0.0)
        return f"{self.last_farm_action} [Q={q_val:.1f}]"

    # ==========================================================================
    # 3. ТАКТИЧЕСКИЙ БОЕВОЙ МОЗГ (ОБУЧЕНИЕ ДУЭЛЯМ И УБИЙСТВУ ГЕРОЕВ)
    # ==========================================================================
    def evaluate_combat_stance(
        self,
        dist_enemy: float,
        can_combo: bool,
        is_turret_near: bool,
        hp_self_ratio: float = 1.0,
    ) -> str:
        """
        Принимает тактическое решение в дуэли через Q-Learning:
        Оценивает дистанцию, свое здоровье и кулдауны способностей.
        """
        # Если вышка врага рядом -> безусловный тактический отход
        if is_turret_near:
            state = "TURRET_DANGER"
            self.last_combat_state = state
            if state not in self.combat_q_table:
                self.combat_q_table[state] = {act: -50.0 for act in self.COMBAT_ACTIONS}
                self.combat_q_table[state]["TURRET_RETREAT"] = 40.0
            self.last_combat_action = "TURRET_RETREAT"
            return "TURRET_RETREAT"

        # Если здоровье Клода критическое -> тактический отход
        if hp_self_ratio < 0.32:
            state = "CRITICAL_HP_DANGER"
            self.last_combat_state = state
            if state not in self.combat_q_table:
                self.combat_q_table[state] = {act: -60.0 for act in self.COMBAT_ACTIONS}
                self.combat_q_table[state]["TACTICAL_RETREAT"] = 50.0
            self.last_combat_action = "TACTICAL_RETREAT"
            return "TACTICAL_RETREAT"

        # Дискретизация дистанции
        if dist_enemy < 260.0:
            dist_bucket = "DANGER_CLOSE"
        elif dist_enemy <= 430.0:
            dist_bucket = "SWEET_SPOT"
        else:
            dist_bucket = "CHASE_FAR"

        hp_cat = "HIGH" if hp_self_ratio >= 0.65 else "MID"
        state = f"{dist_bucket}_{hp_cat}_COMBO_{can_combo}"
        self.last_combat_state = state

        if state not in self.combat_q_table:
            # Начальные эвристики для быстрого обучения
            self.combat_q_table[state] = {
                "KITE_AND_POKE": 35.0 if dist_bucket == "DANGER_CLOSE" else 15.0,
                "SWEET_SPOT_BURST": 40.0 if dist_bucket == "SWEET_SPOT" else 5.0,
                "DIVE_ALL_IN": 50.0 if (dist_bucket == "SWEET_SPOT" and can_combo and hp_cat == "HIGH") else -15.0,
                "TACTICAL_RETREAT": 25.0 if (dist_bucket == "DANGER_CLOSE" and hp_cat == "MID") else -10.0,
            }

        # Выбор действия: вспоминаем победный опыт из лучших моментов (Memory Recall)
        recalled_act = self.recall_best_tactic(state)
        q_vals = self.combat_q_table[state]

        if recalled_act and random.random() < 0.70:
            chosen = recalled_act
        elif random.random() < 0.08:  # 8% исследование новых боевых фишек
            chosen = random.choice(self.DUEL_ACTIONS)
        else:
            max_q = max(q_vals.values())
            best_acts = [act for act, q in q_vals.items() if q == max_q]
            chosen = random.choice(best_acts)

        self.last_combat_action = chosen
        self.log_action("COMBAT", state, chosen)
        return chosen

    def reward_combat(self, reward_amount: float, reason: str = "") -> None:
        """Поощряет или наказывает тактическое решение в бою."""
        if not self.last_combat_state or not self.last_combat_action:
            return

        s = self.last_combat_state
        a = self.last_combat_action

        old_q = self.combat_q_table[s].get(a, 0.0)
        new_q = old_q + self.alpha * (reward_amount - old_q)
        self.combat_q_table[s][a] = round(new_q, 2)

        if abs(reward_amount) >= 25:
            print(f"[🧠 ОБУЧЕНИЕ БОЯ] Тактика '{a}' в '{s}' обновлена до Q={new_q:.1f} ({reason})")
            self.save_memory()

    def get_current_combat_info(self) -> str:
        """Статус текущего решения дуэли для HUD."""
        if not self.last_combat_state or not self.last_combat_action:
            return "COMBAT"
        q_val = self.combat_q_table.get(self.last_combat_state, {}).get(self.last_combat_action, 0.0)
        return f"{self.last_combat_action} [Q={q_val:.1f}]"

    # ==========================================================================
    # 4. БАНК ЛУЧШИХ МОМЕНТОВ И ВОСПОМИНАНИЯ (EXPERIENCE REPLAY MEMORY)
    # ==========================================================================
    def log_action(self, category: str, state: str, action: str) -> None:
        """Сохраняет действие в скользящую историю для связывания с триумфами."""
        self.recent_action_history.append({
            "category": category,
            "state": state,
            "action": action,
            "time": time.time()
        })
        if len(self.recent_action_history) > 30:
            self.recent_action_history = self.recent_action_history[-30:]

    def record_best_moment(
        self,
        moment_type: str,
        reward: float,
        description: str,
        hp_left_pct: int = 100
    ) -> Dict:
        """
        Заносит триумфальный момент в «Золотой фонд» памяти Клода (Highlight Replay Buffer)
        и усиливает Q-веса всей цепочки действий, приведших к триумфу (Experience Replay).
        """
        now = time.time()
        # Извлекаем последние действия за последние 8 секунд
        recent = [a for a in self.recent_action_history if (now - a["time"]) <= 8.0][-6:]
        action_names = [a["action"] for a in recent]

        moment = {
            "id": len(self.best_moments) + 1,
            "type": moment_type,
            "reward": reward,
            "hp_pct": hp_left_pct,
            "actions": action_names,
            "description": description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.best_moments.append(moment)
        self.last_highlight_text = f"[{moment_type}] {description} (+{int(reward)} 🥕)"

        # EXPERIENCE REPLAY: Усиленно прокачиваем Q-веса цепочки действий, приведших к триумфу!
        for item in recent:
            cat = item["category"]
            s = item["state"]
            a = item["action"]
            table = None
            if cat == "COMBAT" and s in self.combat_q_table:
                table = self.combat_q_table
            elif cat == "FARM" and s in self.farm_q_table:
                table = self.farm_q_table
            elif cat == "ROAM" and s in self.roam_q_table:
                table = self.roam_q_table

            if table and s in table and a in table[s]:
                old_q = table[s][a]
                replay_q = old_q + 0.35 * (reward - old_q)
                table[s][a] = round(replay_q, 2)

        print(f"\n========================================================")
        print(f" [🏆 ЗОЛОТОЙ МОМЕНТ СОХРАНЕН В ПАМЯТИ КЛОДА!]")
        print(f" Событие: {moment_type} | Награда: +{reward:.0f} 🥕 | HP: {hp_left_pct}%")
        print(f" Тактическая связка: {' -> '.join(action_names) if action_names else 'Solo Strike'}")
        print(f" Описание: {description}")
        print(f" Всего лучших моментов в голове: {len(self.best_moments)}")
        print(f"========================================================\n")

        self.save_memory()
        return moment

    def recall_best_tactic(self, current_state: str) -> Optional[str]:
        """
        Воспоминание: проверяет сохраненные лучшие моменты на предмет победной тактики.
        """
        if not self.best_moments:
            return None
        # Проверяем последние сохраненные триумфы
        for m in reversed(self.best_moments[-10:]):
            if m.get("actions"):
                for act in m["actions"]:
                    if act in self.COMBAT_ACTIONS:
                        return act
        return None
