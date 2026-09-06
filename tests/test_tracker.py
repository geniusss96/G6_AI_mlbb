"""
Unit tests for world.tracker module.
Verifies track lifecycle, nearest distance association, hysteresis holding,
timeout removal, deterministic timestamps, and reset.
"""

import math
from vision.detector import Detection
from world.tracker import WorldTracker, TrackLifecycle

TOLERANCE = 1e-4


def make_detection(
    cx: float, cy: float, w: float = 40.0, h: float = 40.0,
    class_id: int = 1, class_name: str = "hp_enemy", conf: float = 0.90
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        x1=cx - w / 2.0,
        y1=cy - h / 2.0,
        x2=cx + w / 2.0,
        y2=cy + h / 2.0,
    )


def test_single_detection_creates_track():
    """First detection should spawn a new track."""
    tracker = WorldTracker(lost_timeout_sec=0.9)
    det = make_detection(500.0, 300.0)

    tracks = tracker.update([det], timestamp=10.0)

    assert len(tracks) == 1
    t = tracks[0]
    assert t.track_id == 1
    assert math.isclose(t.position.x, 500.0, rel_tol=TOLERANCE)
    assert math.isclose(t.position.y, 300.0, rel_tol=TOLERANCE)
    assert t.visible is True
    assert t.first_seen == 10.0
    assert t.last_seen == 10.0
    assert t.state == TrackLifecycle.NEW


def test_multiple_detections_distinct_tracks():
    """Multiple distinct detections spawn distinct track IDs."""
    tracker = WorldTracker()
    d1 = make_detection(100.0, 100.0)
    d2 = make_detection(800.0, 400.0)

    tracks = tracker.update([d1, d2], timestamp=1.0)
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}


def test_track_continuity_and_movement():
    """Nearby detection in next frame updates same track ID."""
    tracker = WorldTracker(max_matching_distance_px=50.0)
    d1 = make_detection(500.0, 300.0)
    tracker.update([d1], timestamp=1.0)

    # Move object by 10px (within matching distance)
    d2 = make_detection(510.0, 305.0)
    tracks2 = tracker.update([d2], timestamp=1.1)

    assert len(tracks2) == 1
    t = tracks2[0]
    assert t.track_id == 1  # Preserved ID
    assert math.isclose(t.position.x, 510.0, rel_tol=TOLERANCE)
    assert t.first_seen == 1.0
    assert t.last_seen == 1.1
    assert t.state == TrackLifecycle.VISIBLE


def test_hysteresis_holding_and_timeout():
    """Object temporarily lost is held for lost_timeout, then removed."""
    tracker = WorldTracker(lost_timeout_sec=0.90)
    d = make_detection(500.0, 300.0)

    # Frame 1: Seen at t=1.0
    tracker.update([d], timestamp=1.0)

    # Frame 2: Missing at t=1.5 (< 0.9s timeout) -> Track held, visible=False
    tracks_f2 = tracker.update([], timestamp=1.5)
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == 1
    assert tracks_f2[0].visible is False
    assert tracks_f2[0].state == TrackLifecycle.TEMPORARILY_LOST

    # Frame 3: Missing at t=2.0 (> 0.9s timeout: 2.0 - 1.0 = 1.0 > 0.9) -> Track expired
    tracks_f3 = tracker.update([], timestamp=2.0)
    assert len(tracks_f3) == 0


def test_reappearance_after_timeout_gets_new_id():
    """Object appearing after timeout gets a fresh track ID."""
    tracker = WorldTracker(lost_timeout_sec=0.5)
    d = make_detection(500.0, 300.0)

    tracker.update([d], timestamp=1.0)
    tracker.update([], timestamp=2.0)  # Expired

    tracks = tracker.update([d], timestamp=2.1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 2  # New ID


def test_reset_clears_tracker():
    """Reset clears active tracks and counters."""
    tracker = WorldTracker()
    d = make_detection(500.0, 300.0)
    tracker.update([d], timestamp=1.0)
    assert len(tracker.get_active_tracks()) == 1

    tracker.reset()
    assert len(tracker.get_active_tracks()) == 0

    tracks = tracker.update([d], timestamp=2.0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1  # Reset back to 1


if __name__ == "__main__":
    test_single_detection_creates_track()
    test_multiple_detections_distinct_tracks()
    test_track_continuity_and_movement()
    test_hysteresis_holding_and_timeout()
    test_reappearance_after_timeout_gets_new_id()
    test_reset_clears_tracker()
    print("All world.tracker tests passed successfully!")
