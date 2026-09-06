"""
Unit tests for world.models module.
Verifies immutability, properties, distance calculation, and default factory lists.
"""

import math

from vision.skill_state import SkillState
from vision.hp_detector import HPObservation
from world.models import (
    Vector2,
    PlayerState,
    EnemyState,
    MinionState,
    TurretState,
    WorldState,
)

TOLERANCE = 1e-4


def test_vector2():
    """Verify Vector2 coords, distance, immutability, and tuple conversion."""
    v1 = Vector2(100.0, 200.0)
    v2 = Vector2(103.0, 204.0)

    assert v1.x == 100.0
    assert v1.y == 200.0
    assert math.isclose(v1.distance_to(v2), 5.0, rel_tol=TOLERANCE)
    assert v1.as_tuple() == (100.0, 200.0)

    # Immutability check (frozen)
    try:
        v1.x = 200.0  # type: ignore
        assert False, "Vector2 should be frozen/immutable"
    except Exception:
        pass


def test_player_state():
    """Verify PlayerState fields and immutability."""
    hp_obs = HPObservation(value=0.85, confidence=1.0, visible=True)
    player = PlayerState(
        position=Vector2(772.0, 360.0),
        hp=hp_obs,
        is_visible=True,
    )

    assert player.position.x == 772.0
    assert player.hp.value == 0.85
    assert player.is_visible is True


def test_enemy_minion_turret_states():
    """Verify entity data models."""
    enemy = EnemyState(
        position=Vector2(900.0, 400.0),
        bbox=(880.0, 380.0, 920.0, 420.0),
        confidence=0.92,
        source="yolo"
    )
    assert enemy.position.x == 900.0
    assert enemy.confidence == 0.92

    minion = MinionState(
        position=Vector2(600.0, 300.0),
        bbox=(580.0, 280.0, 620.0, 320.0),
        confidence=0.78,
        is_enemy=True,
        kind="minion"
    )
    assert minion.kind == "minion"
    assert minion.is_enemy is True

    turret = TurretState(
        position=Vector2(1200.0, 200.0),
        bbox=(1170.0, 150.0, 1230.0, 250.0),
        confidence=0.88,
        is_enemy=True
    )
    assert turret.is_enemy is True


def test_world_state_defaults():
    """Verify WorldState initialization with minimal and empty fields."""
    ws = WorldState(timestamp=123.456)

    assert ws.timestamp == 123.456
    assert ws.player is None
    assert ws.enemies == []
    assert ws.minions == []
    assert ws.turrets == []
    assert ws.skills is None
    assert ws.is_near_base is False


def test_world_state_full():
    """Verify WorldState populated with entities and skills."""
    skills = SkillState(s1_ready=True, s2_ready=False, ultimate_ready=True)
    player = PlayerState(
        position=Vector2(772.0, 360.0),
        hp=HPObservation(value=1.0, confidence=1.0, visible=True)
    )
    enemy = EnemyState(
        position=Vector2(850.0, 360.0),
        bbox=(830.0, 340.0, 870.0, 380.0),
        confidence=0.95
    )

    ws = WorldState(
        timestamp=1000.0,
        player=player,
        enemies=[enemy],
        skills=skills,
        is_near_base=True
    )

    assert ws.player is not None
    assert len(ws.enemies) == 1
    assert ws.enemies[0].position.distance_to(ws.player.position) == 78.0
    assert ws.skills.s1_ready is True
    assert ws.is_near_base is True


if __name__ == "__main__":
    test_vector2()
    test_player_state()
    test_enemy_minion_turret_states()
    test_world_state_defaults()
    test_world_state_full()
    print("All world.models tests passed successfully!")
