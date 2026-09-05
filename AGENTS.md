# Claude Tactical AI V2 — Development Rules

## 1. Основное правило
Нельзя выполнять крупный рефакторинг нескольких подсистем одновременно.
Каждая задача изменяет **только одну ответственность**.

## 2. Production Baseline
`realtime_vision.py`, `claude_macro.py`, `joystick_controller.py`, `claude_brain.py` считаются рабочим **V1 baseline**.
Поведение V1 не изменяется без отдельной согласованной задачи.

## 3. Замороженные модули (Legacy)
Не использовать и не расширять:
- `claude_cv_bot.py`
- `combo_macro.py`
- `mlbb_assistant.py`
- `target_attacker.py`
- `humanizer.py`

Они являются историческими/legacy реализациями.

## 4. Архитектура V2
Поток данных строго однонаправленный:
```
VISION → WORLD STATE → TACTICAL BRAIN → ACTION → CONTROL → ADB
```

### Запрещено:
- Brain напрямую вызывает ADB.
- Vision напрямую вызывает Brain.
- World State отправляет команды устройству.
- HUD изменяет состояние Brain.
- Разные модули создают собственные независимые ADB transport.

## 5. Система координат (Coordinate System)
Внутренняя система координат: **1544 × 720**.
Любое преобразование координат выполняется **только** через `vision.geometry`.
Запрещено создавать локальные ad-hoc коэффициенты масштабирования в других модулях.

## 6. Транспорт ADB (Centralized Control)
Все команды устройству проходят через единый `control.adb`.
Запрещено создавать новые прямые `subprocess.run(["adb", ...])` или `subprocess.Popen` в функциональных модулях.

## 7. Обучение с подкреплением (Reinforcement Learning)
Не изменять reward/Q-learning математику во время архитектурной миграции.
Сначала достигается эквивалентность V1/V2.
Любое изменение алгоритма RL выделяется в отдельную задачу.

## 8. Абстракция команд (Action Layer)
Brain возвращает абстрактный `Action` (например: `MOVE`, `ATTACK`, `CAST_S1`, `CAST_S2`, `CAST_ULT`, `RETREAT`, `SCOUT`, `FARM`, `KITE`).
Brain не должен знать конкретные экранные координаты кнопок Android или способ отправки через ADB.

## 9. Дисциплина тестирования (Testing)
После каждого шага миграции:
- Проверка синтаксиса Python (`python -m py_compile ...`).
- Проверка отсутствия циклических импортов.
- Проверка запуска без падений при отсутствии устройства.
- Сравнение поведения с V1 baseline.
- Не удалять legacy-код до завершения полного V2 integration test.

## 10. Дисциплина коммитов (Commit Discipline)
Один архитектурный шаг = один commit.
Примеры:
- `refactor: extract geometry layer`
- `refactor: extract capture service`
- `refactor: introduce WorldState`
- `refactor: introduce Action model`
- `refactor: centralize adb transport`
- `refactor: extract tactical brain`
