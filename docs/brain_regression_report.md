# V1/V2 Brain Regression Report

Автоматический отчёт регрессионного сравнения поведения legacy **V1 Brain** (`ClaudeRLBrain`) и **V2 Brain** (`ClaudeBrainV2`) на одинаковых входных состояниях `TacticalState`.

---

## 1. Сводка (Executive Summary)

- **Total scenarios:** 14
- **Matches:** 14
- **Mismatches:** 0
- **Not comparable:** 0
- **Success Rate:** 100.0%

---

## 2. Детальная Регрессионная Таблица

| # | Сценарий (`BrainScenario`) | Решение V1 (`ClaudeRLBrain`) | Действие V2 (`ActionType`) | Результат (`Result`) | Примечание |
|---|---|:---:|:---:|:---:|---|
| 1 | `scenario_turret_retreat` | `TURRET_RETREAT` | `ActionType.RETREAT` | **MATCH** | Безусловный приоритет вражеской вышки |
| 2 | `scenario_tactical_retreat` | `TACTICAL_RETREAT` | `ActionType.RETREAT` | **MATCH** | Безусловный отход при `hp_ratio < 0.32` |
| 3 | `scenario_kite_and_poke` | `KITE_AND_POKE` | `ActionType.KITE` | **MATCH** | Кайтинг при сближении (`dist < 260.0`) |
| 4 | `scenario_sweet_spot_burst` | `SWEET_SPOT_BURST` | `ActionType.ATTACK` | **MATCH** | Орбитальный стрейф в сладкой зоне (260..430 px) |
| 5 | `scenario_dive_all_in` | `DIVE_ALL_IN` | `ActionType.CAST_ULT` | **MATCH** | Полный врыв при готовности комбо (S2 + Ult) |
| 6 | `scenario_farm_approach` | `FARM_APPROACH` | `ActionType.MOVE` | **MATCH** | Сближение с дальней пачкой крипов (`dist > 420.0`) |
| 7 | `scenario_farm_kite_back` | `FARM_KITE_BACK` | `ActionType.KITE` | **MATCH** | Отход от крипов ближнего боя (`dist < 260.0`) |
| 8 | `scenario_farm_sweet_spot` | `FARM_SWEET_SPOT` | `ActionType.FARM` | **MATCH** | Орб-вокинг в зоне эффективного фарминга |
| 9 | `scenario_farm_s1_aoe` | `FARM_S1_AOE` | `ActionType.CAST_S1` | **MATCH** | АОЕ прокаст S1 по пачке крипов |
| 10 | `scenario_lane_advance` | `LANE_ADVANCE` | `ActionType.MOVE` | **MATCH** | Разведка: продвижение по линии (`time < 5.0`) |
| 11 | `scenario_lane_push` | `LANE_PUSH` | `ActionType.MOVE` | **MATCH** | Разведка: пуш линии (`5.0 <= time < 15.0`) |
| 12 | `scenario_river_scout` | `RIVER_SCOUT` | `ActionType.SCOUT` | **MATCH** | Разведка: контроль реки (`time >= 15.0`) |
| 13 | `scenario_flank_advance` | `FLANK_ADVANCE` | `ActionType.SCOUT` | **MATCH** | Разведка: обход фланга (`time >= 15.0`) |
| 14 | `scenario_creep_intercept` | `CREEP_INTERCEPT` | `ActionType.FARM` | **MATCH** | Разведка: перехват крипов снизу (`time >= 15.0`) |

---

## 3. Гарантии Изоляции и Детерминизма

1. **Изоляция хранилища:**
   - Регрессионный стенд работает с виртуальной памятью `__regression_dummy_isolated__.json`.
   - Файлы `q_brain.json` и `data/q_brain_baseline.json` защищены контрольными суммами SHA-256 (0 изменений).
2. **Детерминизм вывода:**
   - Эксплорация отключена (`exploration_rate = 0.0`), случайность зафиксирована детерминированным генератором.
   - Повторные прогоны дают побитово идентичный результат.
3. **Сохранение метаданных:**
   - `target_id` (для героев и крипов) и векторы направлений `direction` сохраняются без искажений.
