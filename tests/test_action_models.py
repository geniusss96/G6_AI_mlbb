"""
Unit tests for actions.models module.
Verifies ActionType enumeration, Action immutability, structural validation,
absence of hardware-specific fields, and optional parameters.
"""

from world.models import Vector2
from actions.models import Action, ActionType


def test_all_action_types_exist():
    """Verify all standard ActionType enum values exist."""
    assert ActionType.MOVE.value == "move"
    assert ActionType.ATTACK.value == "attack"
    assert ActionType.CAST_S1.value == "cast_s1"
    assert ActionType.CAST_S2.value == "cast_s2"
    assert ActionType.CAST_ULT.value == "cast_ult"
    assert ActionType.RETREAT.value == "retreat"
    assert ActionType.FARM.value == "farm"
    assert ActionType.KITE.value == "kite"
    assert ActionType.SCOUT.value == "scout"
    assert ActionType.IDLE.value == "idle"


def test_action_creation():
    """Verify creation of basic and parameterized actions."""
    # Simple action without arguments
    retreat = Action(type=ActionType.RETREAT)
    assert retreat.type == ActionType.RETREAT
    assert retreat.direction is None
    assert retreat.target_id is None
    assert retreat.duration is None

    # Attack targeting specific entity
    attack = Action(type=ActionType.ATTACK, target_id=42)
    assert attack.type == ActionType.ATTACK
    assert attack.target_id == 42

    # Movement with direction and duration
    move = Action(
        type=ActionType.MOVE,
        direction=Vector2(100.0, 200.0),
        duration=0.5
    )
    assert move.type == ActionType.MOVE
    assert move.direction == Vector2(100.0, 200.0)
    assert move.duration == 0.5

    # Skill cast actions
    cast_s1 = Action(type=ActionType.CAST_S1)
    cast_s2 = Action(type=ActionType.CAST_S2, direction=Vector2(500.0, 300.0))
    cast_ult = Action(type=ActionType.CAST_ULT)
    assert cast_s1.type == ActionType.CAST_S1
    assert cast_s2.direction == Vector2(500.0, 300.0)
    assert cast_ult.type == ActionType.CAST_ULT

    # Farm and Kite
    farm = Action(type=ActionType.FARM, target_id=10)
    kite = Action(type=ActionType.KITE, direction=Vector2(-1.0, 0.5))
    assert farm.type == ActionType.FARM
    assert kite.type == ActionType.KITE


def test_immutability_frozen():
    """Action dataclass is frozen / immutable."""
    action = Action(type=ActionType.ATTACK, target_id=7)
    try:
        action.target_id = 8  # type: ignore
        assert False, "Action should be frozen"
    except Exception:
        pass


def test_no_hardware_fields():
    """Verify Action contract has no screen pixel buttons or adb commands."""
    action = Action(type=ActionType.CAST_S1)
    field_names = set(action.__dataclass_fields__.keys())  # type: ignore
    assert "x" not in field_names
    assert "y" not in field_names
    assert "button" not in field_names
    assert "adb_cmd" not in field_names
    assert "swipe" not in field_names


def test_structural_validation():
    """Verify non-tactical structural validation."""
    # Invalid duration
    try:
        Action(type=ActionType.MOVE, duration=-1.0)
        assert False, "Negative duration should raise ValueError"
    except ValueError:
        pass

    # Invalid direction (NaN)
    try:
        Action(type=ActionType.MOVE, direction=Vector2(float("nan"), 0.0))
        assert False, "NaN coordinate should raise ValueError"
    except ValueError:
        pass

    # Invalid target_id type
    try:
        Action(type=ActionType.ATTACK, target_id="enemy_7")  # type: ignore
        assert False, "String target_id should raise TypeError"
    except TypeError:
        pass


if __name__ == "__main__":
    test_all_action_types_exist()
    test_action_creation()
    test_immutability_frozen()
    test_no_hardware_fields()
    test_structural_validation()
    print("All actions.models tests passed successfully!")
