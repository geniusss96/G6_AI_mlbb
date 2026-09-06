"""Unit and regression tests for brain/transition_builder.py (Stage 17C: Transition Builder)."""

import math
import unittest

from actions.models import Action, ActionType
from brain.encoder import TacticalStateEncoder
from brain.rewards import Reward
from brain.state import TacticalState
from brain.transition import Transition
from brain.transition_builder import TransitionBuilder
from world.models import Vector2


def _make_state(
    timestamp: float = 100.0,
    player_pos: Vector2 = Vector2(772.0, 360.0),
    hp_ratio: float = 1.0,
    is_dead: bool = False,
    s1_ready: bool = True,
    s2_ready: bool = True,
    ult_ready: bool = True,
    can_combo: bool = True,
    enemy_visible: bool = False,
    enemy_count: int = 0,
    enemy_dist: float = None,
    enemy_pos: Vector2 = None,
    enemy_id: int = None,
    minion_count: int = 0,
    minion_dist: float = None,
    minion_pos: Vector2 = None,
    minion_id: int = None,
    turret_near: bool = False,
    near_base: bool = False,
) -> TacticalState:
    return TacticalState(
        timestamp=timestamp,
        player_position=player_pos,
        player_hp_ratio=hp_ratio,
        is_player_dead=is_dead,
        s1_ready=s1_ready,
        s2_ready=s2_ready,
        ultimate_ready=ult_ready,
        can_combo=can_combo,
        enemy_visible=enemy_visible,
        enemy_count=enemy_count,
        nearest_enemy_distance=enemy_dist,
        nearest_enemy_position=enemy_pos,
        minion_count=minion_count,
        nearest_minion_distance=minion_dist,
        nearest_minion_position=minion_pos,
        enemy_near_turret=turret_near,
        is_near_base=near_base,
        nearest_enemy_id=enemy_id,
        nearest_minion_id=minion_id,
    )


class TestTransitionBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.encoder = TacticalStateEncoder()
        self.builder = TransitionBuilder(self.encoder)

    def test_non_terminal_transition_build(self) -> None:
        """Verifies standard non-terminal transition construction and encoding."""
        s_curr = _make_state(enemy_visible=True, enemy_dist=200.0, hp_ratio=0.80, can_combo=True)
        s_next = _make_state(enemy_visible=True, enemy_dist=350.0, hp_ratio=0.75, can_combo=True)
        action = Action(type=ActionType.KITE)

        trans = self.builder.build(
            state=s_curr,
            action=action,
            reward=15.0,
            next_state=s_next,
            done=False,
        )

        self.assertIsInstance(trans, Transition)
        self.assertEqual(trans.state_key, "DANGER_CLOSE_HIGH_COMBO_True")
        self.assertEqual(trans.action, "kite")
        self.assertEqual(trans.reward, 15.0)
        self.assertEqual(trans.next_state_key, "SWEET_SPOT_HIGH_COMBO_True")
        self.assertFalse(trans.done)

    def test_terminal_transition_with_none_next_state(self) -> None:
        """When done=True, next_state can be None and next_state_key becomes None."""
        s_curr = _make_state(enemy_visible=True, enemy_dist=200.0, hp_ratio=0.20)
        action = Action(type=ActionType.RETREAT)

        trans = self.builder.build(
            state=s_curr,
            action=action,
            reward=-40.0,
            next_state=None,
            done=True,
        )

        self.assertIsInstance(trans, Transition)
        self.assertEqual(trans.state_key, "CRITICAL_HP_DANGER")
        self.assertEqual(trans.action, "retreat")
        self.assertEqual(trans.reward, -40.0)
        self.assertIsNone(trans.next_state_key)
        self.assertTrue(trans.done)

    def test_terminal_transition_with_next_state_present(self) -> None:
        """When done=True and next_state is provided, next_state_key is encoded."""
        s_curr = _make_state(enemy_visible=True, enemy_dist=300.0, hp_ratio=0.70, can_combo=True)
        s_next = _make_state(is_dead=True, enemy_visible=False)
        action = Action(type=ActionType.CAST_ULT)

        trans = self.builder.build(
            state=s_curr,
            action=action,
            reward=60.0,
            next_state=s_next,
            done=True,
        )

        self.assertEqual(trans.state_key, "SWEET_SPOT_HIGH_COMBO_True")
        self.assertEqual(trans.action, "cast_ult")
        self.assertTrue(trans.done)
        self.assertIsNotNone(trans.next_state_key)

    def test_done_false_with_none_next_state_raises_value_error(self) -> None:
        """When done=False, omitting next_state must raise ValueError."""
        s_curr = _make_state(enemy_visible=True, enemy_dist=300.0)
        action = Action(type=ActionType.ATTACK)

        with self.assertRaises(ValueError):
            self.builder.build(
                state=s_curr,
                action=action,
                reward=10.0,
                next_state=None,
                done=False,
            )

    def test_invalid_state_types_rejected(self) -> None:
        """Non-TacticalState objects must raise TypeError."""
        action = Action(type=ActionType.MOVE)
        s_valid = _make_state()

        with self.assertRaises(TypeError):
            self.builder.build("not_a_state", action, 0.0, s_valid, False)  # type: ignore

        with self.assertRaises(TypeError):
            self.builder.build(s_valid, action, 0.0, "not_a_next_state", False)  # type: ignore

    def test_invalid_action_types_and_empty_strings(self) -> None:
        """Invalid actions must raise TypeError or ValueError."""
        s = _make_state()
        with self.assertRaises(TypeError):
            self.builder.build(s, 12345, 0.0, s, False)  # type: ignore

        with self.assertRaises(ValueError):
            self.builder.build(s, "", 0.0, s, False)

    def test_reward_object_and_nan_inf_validation(self) -> None:
        """Reward object is unwrapped to float; NaN/Inf are rejected."""
        s = _make_state()
        action = Action(type=ActionType.ATTACK)

        # Reward dataclass support
        reward_obj = Reward(value=-25.0, reason="wall collision", source="tactical")
        trans = self.builder.build(s, action, reward_obj, s, False)
        self.assertEqual(trans.reward, -25.0)

        # NaN rejection
        with self.assertRaises(ValueError):
            self.builder.build(s, action, float("nan"), s, False)

        # Inf rejection
        with self.assertRaises(ValueError):
            self.builder.build(s, action, float("inf"), s, False)

    def test_deterministic_output(self) -> None:
        """Repeated calls with identical parameters yield identical Transitions."""
        s_a = _make_state(enemy_visible=True, enemy_dist=300.0, hp_ratio=0.80)
        s_b = _make_state(enemy_visible=True, enemy_dist=200.0, hp_ratio=0.75)
        act = Action(type=ActionType.KITE)

        t1 = self.builder.build(s_a, act, -5.0, s_b, False)
        t2 = self.builder.build(s_a, act, -5.0, s_b, False)

        self.assertEqual(t1, t2)
        self.assertEqual(hash(t1), hash(t2))

    def test_regression_scenario_state_a_kite_reward_state_b(self) -> None:
        """Regression specification from prompt:

        TacticalState A -> Action KITE -> reward -5 -> TacticalState B -> Transition.
        Verifies expected stable state keys for A and B.
        """
        state_a = _make_state(
            enemy_visible=True,
            enemy_count=1,
            enemy_dist=350.0,  # SWEET_SPOT
            hp_ratio=0.70,     # HIGH
            can_combo=False,
        )
        state_b = _make_state(
            enemy_visible=True,
            enemy_count=1,
            enemy_dist=240.0,  # DANGER_CLOSE (< 260)
            hp_ratio=0.68,     # HIGH
            can_combo=False,
        )
        action_kite = Action(type=ActionType.KITE)

        transition = self.builder.build(
            state=state_a,
            action=action_kite,
            reward=-5.0,
            next_state=state_b,
            done=False,
        )

        self.assertEqual(transition.state_key, "SWEET_SPOT_HIGH_COMBO_False")
        self.assertEqual(transition.action, "kite")
        self.assertEqual(transition.reward, -5.0)
        self.assertEqual(transition.next_state_key, "DANGER_CLOSE_HIGH_COMBO_False")
        self.assertFalse(transition.done)

    def test_architecture_isolation(self) -> None:
        """brain.transition_builder must have zero dependencies on hardware, vision, or ADB."""
        import brain.transition_builder as tb_mod
        forbidden = [
            "subprocess",
            "cv2",
            "control",
            "adb",
            "vision",
            "q_learning",
            "realtime_vision",
        ]
        for f in forbidden:
            self.assertFalse(hasattr(tb_mod, f))


if __name__ == "__main__":
    unittest.main()
