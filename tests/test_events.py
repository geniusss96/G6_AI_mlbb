"""
Unit tests for world.events Event layer.
Verifies GameEvent models, WorldEventDetector, distinct detection loss vs kill,
deduplication, player death, and creep kills on deterministic timestamps.
"""

from world.models import Vector2, WorldState, PlayerState
from world.tracker import Track
from world.events import EventType, GameEvent, WorldEventDetector, WatchdogManager
from vision.hp_detector import HPObservation


def make_track(tid: int, x: float, y: float, name: str = "hp_enemy", visible: bool = True) -> Track:
    return Track(
        track_id=tid,
        position=Vector2(x, y),
        bbox=(x - 20, y - 20, x + 20, y + 20),
        confidence=0.90,
        class_name=name,
        class_id=1,
        visible=visible,
        first_seen=0.0,
        last_seen=0.0,
    )


def test_target_entered_and_lost():
    """Verify TARGET_ENTERED and TARGET_LOST observation events without declaring death."""
    detector = WorldEventDetector()
    ws = WorldState(timestamp=1.0)

    t1 = make_track(1, 500.0, 300.0, visible=True)
    events1 = detector.update(ws, [t1], timestamp=1.0)
    assert len(events1) == 1
    assert events1[0].type == EventType.TARGET_ENTERED
    assert events1[0].entity_id == 1

    # Next frame track is lost
    events2 = detector.update(ws, [], timestamp=1.1)
    assert len(events2) == 1
    assert events2[0].type == EventType.TARGET_LOST
    assert events2[0].entity_id == 1
    # CRITICAL: TARGET_LOST != HERO_KILL!
    assert not any(e.type == EventType.HERO_KILL for e in events2)


def test_hero_disappearance_without_combat_is_not_kill():
    """
    CRITICAL REGRESSION TEST:
    Enemy visible -> enemy leaves frame / becomes lost without combat -> NO HERO_KILL!
    """
    detector = WorldEventDetector()
    ws = WorldState(timestamp=1.0)

    t = make_track(1, 500.0, 300.0, name="hp_enemy")
    detector.update(ws, [t], timestamp=1.0)

    # Enemy simply disappears at t=2.0 (no attack recorded)
    detector.update(ws, [], timestamp=2.0)
    # 3.0s elapsed without enemy
    events = detector.update(ws, [], timestamp=5.0)

    # No HERO_KILL event generated!
    assert not any(e.type == EventType.HERO_KILL for e in events)


def test_hero_kill_confirmed_with_combat_duration():
    """
    Confirmed combat:
    Active combat >= 2.0s + fresh attack <= 3.5s + enemy disappeared >= 2.5s -> EXACTLY ONE HERO_KILL!
    """
    detector = WorldEventDetector()
    ws = WorldState(timestamp=10.0)
    enemy_pos = Vector2(500.0, 300.0)
    t = make_track(1, enemy_pos.x, enemy_pos.y, name="hp_enemy")

    # Start combat at t=10.0
    detector.update(ws, [t], timestamp=10.0)
    detector.record_attack_action(enemy_pos, timestamp=10.0, is_hero=True)

    # Attack ongoing at t=12.2 (combat duration = 2.2s >= 2.0s)
    detector.update(ws, [t], timestamp=12.2)
    detector.record_attack_action(enemy_pos, timestamp=12.2, is_hero=True)

    # Enemy disappears at t=12.3 (HP dropped/slain)
    detector.update(ws, [], timestamp=12.3)

    # After 2.6s of absence (t=14.9, elapsed 2.6s >= 2.5s)
    events = detector.update(ws, [], timestamp=14.9)

    hero_kills = [e for e in events if e.type == EventType.HERO_KILL]
    assert len(hero_kills) == 1
    assert hero_kills[0].type == EventType.HERO_KILL
    assert hero_kills[0].confidence >= 0.90

    # DEDUPLICATION CHECK: Next frames must NOT emit duplicate HERO_KILL
    next_events = detector.update(ws, [], timestamp=15.0)
    assert not any(e.type == EventType.HERO_KILL for e in next_events)


def test_creep_kill_confirmed():
    """Creep continuously attacked >= 2.5s and disappeared >= 2.0s triggers CREEP_KILL."""
    detector = WorldEventDetector()
    ws = WorldState(timestamp=1.0)
    creep_pos = Vector2(400.0, 250.0)
    t = make_track(5, creep_pos.x, creep_pos.y, name="minion_enemy")

    # Start attack at t=1.0
    detector.update(ws, [t], timestamp=1.0)
    detector.record_attack_action(creep_pos, timestamp=1.0, is_hero=False)

    # Attack until t=3.6 (duration 2.6s >= 2.5s)
    detector.update(ws, [t], timestamp=3.6)
    detector.record_attack_action(creep_pos, timestamp=3.6, is_hero=False)

    # Creep disappears at t=3.7
    detector.update(ws, [], timestamp=3.7)

    # After 2.1s (t=5.8 >= 3.7 + 2.0s)
    events = detector.update(ws, [], timestamp=5.8)
    creep_kills = [e for e in events if e.type == EventType.CREEP_KILL]
    assert len(creep_kills) == 1


def test_player_death_event():
    """Player with low HP disappearing during combat triggers PLAYER_DEATH."""
    detector = WorldEventDetector()

    # Player seen with 15% HP at t=10.0 in combat
    player_low_hp = PlayerState(
        position=Vector2(772.0, 360.0),
        hp=HPObservation(value=0.15, confidence=1.0, visible=True),
        is_visible=True,
    )
    ws1 = WorldState(timestamp=10.0, player=player_low_hp)
    detector.record_attack_action(Vector2(800.0, 360.0), timestamp=10.0, is_hero=True)
    detector.update(ws1, [], timestamp=10.0)

    # Player disappears from screen (death animation / respawn screen)
    ws_no_player = WorldState(timestamp=10.5, player=None)
    detector.update(ws_no_player, [], timestamp=10.5)

    # 2.6s elapsed since disappearance (>2.5s timeout)
    events = detector.update(WorldState(timestamp=13.2, player=None), [], timestamp=13.2)
    player_deaths = [e for e in events if e.type == EventType.PLAYER_DEATH]
    assert len(player_deaths) == 1

    # Deduplication: next update does NOT trigger duplicate death
    events2 = detector.update(WorldState(timestamp=13.3, player=None), [], timestamp=13.3)
    assert not any(e.type == EventType.PLAYER_DEATH for e in events2)


if __name__ == "__main__":
    test_target_entered_and_lost()
    test_hero_disappearance_without_combat_is_not_kill()
    test_hero_kill_confirmed_with_combat_duration()
    test_creep_kill_confirmed()
    test_player_death_event()
    print("All world.events tests passed successfully!")
