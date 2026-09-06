"""
tests/test_stage_20_control.py — Stage 20 V2 Control Layer Migration Tests.

Verifies:
1. Concrete implementations of JoystickPort (JoystickControl), SkillsPort (SkillsControl),
   and AttackPort (AttackControl).
2. ADBTransport command routing: all hardware commands flow through ADBTransport without
   creating adb.exe subprocesses for individual actions.
3. Joystick behavior preservation:
   - motionevent DOWN / MOVE / UP lifecycle
   - Screen touch watchdog: notify_screen_touched() resets is_holding
   - Re-issue of DOWN on next movement frame after screen touch
   - 1.8-second watchdog refresh
   - release() behavior
4. Attack behavior preservation:
   - Cooldown pacing (prevents duplicate taps within cooldown window)
   - Coordinate humanization and capacitive hold duration
   - Screen touch notification to joystick
5. Skills behavior preservation:
   - Individual skill casts (S1, S2, basic attack)
   - Panic retreat S2 double-tap
   - True Claude Combo (S2 -> S2 -> S1 -> Ult) strictly 4 steps; NO 5th S2-back step
6. ActionExecutor integration with concrete V2 control classes.
7. Dependency boundary: no hardware/control dependencies in brain; no subprocess in actions.
"""

import ast
import os
import sys
import unittest
from typing import List, Optional

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from control.interfaces import JoystickPort, SkillsPort, AttackPort
from control.joystick import JoystickControl
from control.attack import AttackControl
from control.skills import SkillsControl
from control.humanizer import InputHumanizer
from actions.models import Action, ActionType
from actions.executor import ActionExecutor
from world.models import Vector2


class FakeADBTransport:
    """Mock ADBTransport that records all dispatched commands without hardware access."""

    def __init__(self):
        self.sent_commands: List[str] = []
        self.taps: List[tuple] = []
        self.swipes: List[tuple] = []
        self.motionevents: List[tuple] = []

    def send(self, command: str) -> bool:
        self.sent_commands.append(command)
        return True

    def send_tap(self, x: int, y: int) -> bool:
        self.taps.append((x, y))
        self.sent_commands.append(f"input tap {x} {y}")
        return True

    def send_swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 100) -> bool:
        self.swipes.append((x1, y1, x2, y2, duration_ms))
        self.sent_commands.append(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
        return True

    def send_motionevent(self, action: str, x: int, y: int) -> bool:
        self.motionevents.append((action, x, y))
        self.sent_commands.append(f"input motionevent {action} {x} {y}")
        return True

    def is_alive(self) -> bool:
        return True

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestStage20Control(unittest.TestCase):
    def setUp(self):
        self.transport = FakeADBTransport()
        self.current_time = 100.0
        self.time_provider = lambda: self.current_time

        self.joystick = JoystickControl(
            transport=self.transport,  # type: ignore
            center_x=288,
            center_y=560,
            radius=110,
            time_provider=self.time_provider,
        )
        self.attack_ctrl = AttackControl(
            transport=self.transport,  # type: ignore
            joystick=self.joystick,
            cooldown_sec=0.20,
            time_provider=self.time_provider,
        )
        self.skills_ctrl = SkillsControl(
            transport=self.transport,  # type: ignore
            joystick=self.joystick,
        )

    # --------------------------------------------------------------------------
    # 1. Port Implementation Verification
    # --------------------------------------------------------------------------
    def test_implements_conceptual_ports(self):
        """Concrete control classes satisfy runtime-checkable Protocols."""
        self.assertIsInstance(self.joystick, JoystickPort)
        self.assertIsInstance(self.skills_ctrl, SkillsPort)
        self.assertIsInstance(self.attack_ctrl, AttackPort)

    # --------------------------------------------------------------------------
    # 2. Joystick Lifecycle and Watchdog Tests
    # --------------------------------------------------------------------------
    def test_joystick_initial_down_and_move(self):
        """Initial movement sends DOWN at center, followed by MOVE to target."""
        hero = (772.0, 360.0)
        target = (900.0, 360.0)  # Move strictly right

        joy_pos = self.joystick.move_towards(hero, target)
        self.assertTrue(self.joystick.is_holding)

        # Expected: DOWN at (288, 560), then MOVE at target joy coord
        self.assertEqual(len(self.transport.motionevents), 2)
        down_event = self.transport.motionevents[0]
        move_event = self.transport.motionevents[1]

        self.assertEqual(down_event, ("DOWN", 288, 560))
        self.assertEqual(move_event[0], "MOVE")
        self.assertEqual(move_event[1], 288 + 110)  # angle 0 -> cos = 1 -> 398
        self.assertEqual(move_event[2], 560)
        self.assertEqual(joy_pos, (398, 560))

    def test_joystick_subsequent_move_maintains_hold(self):
        """Subsequent moves while holding (<1.8s) send only MOVE with toggle jitter."""
        hero = (772.0, 360.0)
        target = (900.0, 360.0)

        # First move
        self.joystick.move_towards(hero, target)
        self.assertEqual(len(self.transport.motionevents), 2)

        # Second move 0.1s later
        self.current_time += 0.1
        self.joystick.move_towards(hero, target)

        # Only one additional MOVE event was sent (no new DOWN)
        self.assertEqual(len(self.transport.motionevents), 3)
        self.assertEqual(self.transport.motionevents[2][0], "MOVE")

    def test_screen_touch_resets_holding_and_reissues_down(self):
        """
        Critical V1 semantics:
        Screen touch resets is_holding. Next movement frame must re-issue DOWN
        to prevent Android joystick paralysis.
        """
        hero = (772.0, 360.0)
        target = (900.0, 360.0)

        # 1. Start moving
        self.joystick.move_towards(hero, target)
        self.assertTrue(self.joystick.is_holding)
        self.assertEqual(len(self.transport.motionevents), 2)

        # 2. Screen touch occurs (attack or skill)
        self.joystick.notify_screen_touched()
        self.assertFalse(self.joystick.is_holding)

        # 3. Next movement frame
        self.current_time += 0.05
        self.joystick.move_towards(hero, target)

        # Re-issues DOWN then MOVE!
        self.assertTrue(self.joystick.is_holding)
        self.assertEqual(len(self.transport.motionevents), 4)
        self.assertEqual(self.transport.motionevents[2], ("DOWN", 288, 560))
        self.assertEqual(self.transport.motionevents[3][0], "MOVE")

    def test_watchdog_1_8_seconds_refresh(self):
        """When holding for > 1.8 seconds, watchdog sends fresh DOWN to prevent driver sleep."""
        hero = (772.0, 360.0)
        target = (900.0, 360.0)

        self.joystick.move_towards(hero, target)
        self.assertEqual(len(self.transport.motionevents), 2)

        # Advance time by 2.0 seconds (> 1.8s watchdog threshold)
        self.current_time += 2.0
        self.joystick.move_towards(hero, target)

        # Watchdog re-triggers DOWN and MOVE
        self.assertEqual(len(self.transport.motionevents), 4)
        self.assertEqual(self.transport.motionevents[2], ("DOWN", 288, 560))
        self.assertEqual(self.transport.motionevents[3][0], "MOVE")

    def test_joystick_release_sends_up(self):
        """release() lifts finger from screen (sends UP) and resets state."""
        hero = (772.0, 360.0)
        target = (900.0, 360.0)
        self.joystick.move_towards(hero, target)

        self.assertTrue(self.joystick.is_holding)
        res = self.joystick.release()
        self.assertTrue(res)
        self.assertFalse(self.joystick.is_holding)
        self.assertEqual(self.transport.motionevents[-1][0], "UP")

        # Second release when not holding is safe no-op
        res2 = self.joystick.release()
        self.assertTrue(res2)

    def test_kite_away_opposite_vector(self):
        """kite_away directs stick in opposite direction of danger."""
        hero = (500.0, 300.0)
        danger = (600.0, 300.0)  # Danger is to the right (+100)

        # Kiting away should move stick to the left (-110 from center)
        joy_pos = self.joystick.kite_away(hero, danger)
        self.assertEqual(joy_pos[0], 288 - 110)
        self.assertEqual(joy_pos[1], 560)

    # --------------------------------------------------------------------------
    # 3. Attack Controller Tests
    # --------------------------------------------------------------------------
    def test_attack_cooldown_and_dispatch(self):
        """Attack controller respects cooldown pacing and notifies joystick."""
        self.joystick.move_towards((772.0, 360.0), (800.0, 360.0))
        self.assertTrue(self.joystick.is_holding)

        # First attack succeeds
        res1 = self.attack_ctrl.attack(target_id=10)
        self.assertTrue(res1)
        self.assertEqual(len(self.transport.swipes), 1)
        # Joystick holding was reset by screen touch
        self.assertFalse(self.joystick.is_holding)

        # Immediate second attack (within 0.20s cooldown) is suppressed
        self.current_time += 0.05
        res2 = self.attack_ctrl.attack(target_id=10)
        self.assertFalse(res2)
        self.assertEqual(len(self.transport.swipes), 1)

        # Attack after cooldown window (0.25s) succeeds
        self.current_time += 0.20
        res3 = self.attack_ctrl.attack(target_id=10)
        self.assertTrue(res3)
        self.assertEqual(len(self.transport.swipes), 2)

    # --------------------------------------------------------------------------
    # 4. Skills Controller Tests
    # --------------------------------------------------------------------------
    def test_skill_casts_and_coordinates(self):
        """Skills controller dispatches S1, S2, and basic attack near calibrated coordinates."""
        # S1: nominal (1088, 540)
        self.skills_ctrl.fire_skill_1_fast()
        swipe_s1 = self.transport.swipes[-1]
        self.assertAlmostEqual(swipe_s1[0], 1088, delta=10)
        self.assertAlmostEqual(swipe_s1[1], 540, delta=10)

        # S2: nominal (1228, 392)
        self.skills_ctrl.fire_skill_2_fast()
        swipe_s2 = self.transport.swipes[-1]
        self.assertAlmostEqual(swipe_s2[0], 1228, delta=10)
        self.assertAlmostEqual(swipe_s2[1], 392, delta=10)

        # Basic attack: nominal (1350, 560)
        self.skills_ctrl.fire_basic_attack_fast()
        swipe_atk = self.transport.swipes[-1]
        self.assertAlmostEqual(swipe_atk[0], 1350, delta=10)
        self.assertAlmostEqual(swipe_atk[1], 560, delta=10)

    def test_panic_retreat_s2_double_tap(self):
        """panic_retreat_s2 dispatches chained double-tap on S2."""
        self.skills_ctrl.panic_retreat_s2()
        cmd = self.transport.sent_commands[-1]
        # Must chain two swipes
        swipes_in_cmd = [part for part in cmd.split(" && ") if part.startswith("input swipe")]
        self.assertEqual(len(swipes_in_cmd), 2)

    def test_true_claude_combo_strictly_4_steps(self):
        """
        Preserves true Claude engagement combo:
        S2 (Place) -> S2 (Teleport) -> S1 (Steal) -> Ultimate (Burst)
        Strictly 4 steps: confirms NO 5th S2-back step.
        """
        self.skills_ctrl.trigger_engage_ultimate(ult_channel_ms=1000.0)
        cmd = self.transport.sent_commands[-1]

        steps = [part.strip() for part in cmd.split(" && ") if part.strip().startswith("input swipe")]
        self.assertEqual(
            len(steps),
            4,
            f"True Claude Combo must have strictly 4 steps, found {len(steps)}: {steps}"
        )

        delays = [part.strip() for part in cmd.split(" && ") if part.strip().startswith("usleep")]
        self.assertEqual(len(delays), 3, "4-step combo must have exactly 3 inter-action delays")

    # --------------------------------------------------------------------------
    # 5. Full ActionExecutor Integration with V2 Control
    # --------------------------------------------------------------------------
    def test_action_executor_integrated_with_v2_control(self):
        """ActionExecutor dispatches all Action types cleanly through V2 control implementations."""
        executor = ActionExecutor(
            joystick=self.joystick,
            skills=self.skills_ctrl,
            attack_controller=self.attack_ctrl,
            hero_pos_provider=lambda: (772.0, 360.0),
        )

        actions_to_test = [
            Action(type=ActionType.MOVE, direction=Vector2(850.0, 400.0)),
            Action(type=ActionType.ATTACK, target_id=5),
            Action(type=ActionType.CAST_S1),
            Action(type=ActionType.CAST_S2),
            Action(type=ActionType.CAST_ULT),
            Action(type=ActionType.KITE, direction=Vector2(900.0, 360.0)),
            Action(type=ActionType.RETREAT, direction=Vector2(100.0, 360.0)),
            Action(type=ActionType.FARM, target_id=7, direction=Vector2(800.0, 350.0)),
            Action(type=ActionType.SCOUT, direction=Vector2(600.0, 200.0)),
            Action(type=ActionType.IDLE),
        ]

        for action in actions_to_test:
            self.current_time += 0.5  # clear attack cooldown
            res = executor.execute(action)
            self.assertTrue(res.success, f"Failed to execute action {action.type}: {res.error}")

    # --------------------------------------------------------------------------
    # 6. Static Architecture & Boundary Verification
    # --------------------------------------------------------------------------
    def test_brain_does_not_depend_on_control(self):
        """brain/ package must NOT import control package or hardware modules."""
        brain_dir = os.path.join(WORKSPACE_ROOT, "brain")
        for root, _, files in os.walk(brain_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            root_mod = alias.name.split(".")[0]
                            self.assertNotEqual(
                                root_mod,
                                "control",
                                f"{file} illegally imports control package: {alias.name}"
                            )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            root_mod = node.module.split(".")[0]
                            self.assertNotEqual(
                                root_mod,
                                "control",
                                f"{file} illegally imports from control package: {node.module}"
                            )

    def test_no_direct_subprocess_in_actions(self):
        """actions/ must not spawn subprocesses directly (must use control layer)."""
        actions_dir = os.path.join(WORKSPACE_ROOT, "actions")
        for root, _, files in os.walk(actions_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotEqual(alias.name.split(".")[0], "subprocess")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.assertNotEqual(node.module.split(".")[0], "subprocess")


if __name__ == "__main__":
    unittest.main()
