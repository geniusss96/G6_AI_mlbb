"""
Unit tests for world.events module.
Tests Anti-Corpse Watchdog, Ally-Creep Watchdog, suppression zones,
movement reset, expiry, deterministic timestamps, and "lost != dead" assertion.
"""

from world.models import Vector2
from world.tracker import Track, TrackLifecycle
from world.events import WatchdogManager, SuppressionZone


def make_track(tid: int, x: float, y: float, name: str = "hp_enemy") -> Track:
    return Track(
        track_id=tid,
        position=Vector2(x, y),
        bbox=(x - 20, y - 20, x + 20, y + 20),
        confidence=0.90,
        class_name=name,
        class_id=1,
        visible=True,
        first_seen=0.0,
        last_seen=0.0,
    )


def test_anti_corpse_timer_and_timeout():
    """Stationary target for >2.5s triggers anti-corpse suppression zone."""
    wm = WatchdogManager()
    pos = Vector2(500.0, 300.0)

    # Initial detection at t=1.0
    assert wm.update_enemy_stationary(pos, timestamp=1.0) is False

    # Motionless at t=2.0 (elapsed 1.0s < 2.5s)
    assert wm.update_enemy_stationary(pos, timestamp=2.0) is False
    assert wm.is_suppressed(pos) is False

    # Motionless at t=3.6 (elapsed 2.6s > 2.5s) -> TIMEOUT!
    assert wm.update_enemy_stationary(pos, timestamp=3.6) is True
    assert wm.is_suppressed(pos) is True

    # Check suppression expiration (expiry = 3.6 + 20.0 = 23.6)
    assert wm.is_suppressed(pos) is True
    wm.clean_expired(timestamp=23.7)
    assert wm.is_suppressed(pos) is False


def test_anti_corpse_movement_resets_timer():
    """Target moving >20px resets stationary timer."""
    wm = WatchdogManager()
    pos1 = Vector2(500.0, 300.0)

    wm.update_enemy_stationary(pos1, timestamp=1.0)
    wm.update_enemy_stationary(pos1, timestamp=3.0)  # Elapsed 2.0s

    # Moved by 30px at t=3.2
    pos2 = Vector2(530.0, 300.0)
    assert wm.update_enemy_stationary(pos2, timestamp=3.2) is False

    # At t=4.5 (elapsed 1.3s from movement, not 3.5s) -> NOT timed out
    assert wm.update_enemy_stationary(pos2, timestamp=4.5) is False
    assert wm.is_suppressed(pos2) is False


def test_ally_creep_watchdog():
    """Creep attacked continuously for >4.5s gets blacklisted for 20s."""
    wm = WatchdogManager()
    creep_pos = Vector2(600.0, 400.0)

    # Start attack at t=10.0
    assert wm.update_creep_attack(creep_pos, is_attacking=True, timestamp=10.0) is False

    # Still attacking at t=14.0 (elapsed 4.0s < 4.5s)
    assert wm.update_creep_attack(creep_pos, is_attacking=True, timestamp=14.0) is False

    # Still attacking at t=14.6 (elapsed 4.6s > 4.5s) -> TIMEOUT!
    assert wm.update_creep_attack(creep_pos, is_attacking=True, timestamp=14.6) is True
    assert wm.is_suppressed(creep_pos) is True


def test_filter_suppressed_tracks():
    """Tracks falling inside suppression zone are filtered out."""
    wm = WatchdogManager()
    blacklisted_pos = Vector2(500.0, 300.0)
    wm.update_enemy_stationary(blacklisted_pos, timestamp=1.0)
    wm.update_enemy_stationary(blacklisted_pos, timestamp=4.0)  # Triggers suppression

    t1 = make_track(1, 510.0, 310.0)  # Near corpse (<60px)
    t2 = make_track(2, 800.0, 400.0)  # Safe distant track

    filtered = wm.filter_suppressed_tracks([t1, t2])
    assert len(filtered) == 1
    assert filtered[0].track_id == 2


def test_lost_is_not_dead_distinction():
    """
    Critical requirement: track loss / disappearance does NOT automatically
    trigger corpse suppression or death without watchdog conditions.
    """
    wm = WatchdogManager()
    pos = Vector2(500.0, 300.0)

    # Entity is seen briefly then lost
    wm.update_enemy_stationary(pos, timestamp=1.0)
    # Entity disappears (target_pos is None)
    result = wm.update_enemy_stationary(None, timestamp=1.5)

    assert result is False
    assert wm.is_suppressed(pos) is False
    assert len(wm.get_active_suppression_zones()) == 0


def test_reset():
    """Reset clears all active zones."""
    wm = WatchdogManager()
    pos = Vector2(500.0, 300.0)
    wm.update_enemy_stationary(pos, timestamp=1.0)
    wm.update_enemy_stationary(pos, timestamp=4.0)
    assert wm.is_suppressed(pos) is True

    wm.reset()
    assert wm.is_suppressed(pos) is False
    assert len(wm.get_active_suppression_zones()) == 0


if __name__ == "__main__":
    test_anti_corpse_timer_and_timeout()
    test_anti_corpse_movement_resets_timer()
    test_ally_creep_watchdog()
    test_filter_suppressed_tracks()
    test_lost_is_not_dead_distinction()
    test_reset()
    print("All world.events watchdog tests passed successfully!")
