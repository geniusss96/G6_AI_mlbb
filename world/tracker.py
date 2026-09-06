"""
world.tracker — Discrete object tracking and temporal hysteresis layer.

Maintains track identity across consecutive frames via nearest-neighbor distance matching.
Handles object flicker/occlusion using configurable hysteresis timeouts without
tactical logic, combat state, or death detection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import math

from world.models import Vector2
from vision.detector import Detection


class TrackLifecycle(str, Enum):
    """
    Pure lifecycle states of an actively tracked entity.
    """
    NEW = "NEW"
    VISIBLE = "VISIBLE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    REMOVED = "REMOVED"


@dataclass
class Track:
    """
    Discrete temporal track representing an entity observed across frames.
    """
    track_id: int
    position: Vector2
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    class_id: int

    visible: bool
    first_seen: float
    last_seen: float
    state: TrackLifecycle = TrackLifecycle.NEW

    @property
    def center(self) -> Tuple[float, float]:
        return self.position.as_tuple()


class WorldTracker:
    """
    Deterministic distance-based tracker with hysteresis memory for V2 pipeline.
    """

    def __init__(
        self,
        max_matching_distance_px: float = 75.0,
        lost_timeout_sec: float = 0.90,
    ):
        """
        Args:
            max_matching_distance_px: Max distance between previous position
                                      and new detection to consider same track.
            lost_timeout_sec: Hysteresis duration (default 0.9s matches V1 hysteresis).
        """
        self.max_matching_distance = max_matching_distance_px
        self.lost_timeout = lost_timeout_sec
        self._next_track_id: int = 1
        self._tracks: dict[int, Track] = {}

    def update(
        self,
        detections: List[Detection],
        timestamp: float,
    ) -> List[Track]:
        """
        Updates active tracks with detections from current frame.

        Args:
            detections: Detections in reference resolution (1544x720).
            timestamp: Deterministic monotonic frame timestamp.

        Returns:
            List of currently active tracks (both VISIBLE and TEMPORARILY_LOST within timeout).
        """
        # Group detections by position
        unmatched_detections = list(detections)
        matched_track_ids = set()

        # Step 1: Match existing tracks to nearest compatible detections
        # Sort tracks by last_seen to prioritize freshest observations
        sorted_track_ids = sorted(
            self._tracks.keys(),
            key=lambda tid: self._tracks[tid].last_seen,
            reverse=True
        )

        for track_id in sorted_track_ids:
            track = self._tracks[track_id]
            best_det_idx: Optional[int] = None
            best_dist: float = self.max_matching_distance

            for i, det in enumerate(unmatched_detections):
                # Only match same class or similar category
                if det.class_name != track.class_name:
                    continue

                det_pos = Vector2(det.center[0], det.center[1])
                dist = track.position.distance_to(det_pos)

                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = i

            if best_det_idx is not None:
                matched_det = unmatched_detections.pop(best_det_idx)
                track.position = Vector2(matched_det.center[0], matched_det.center[1])
                track.bbox = (matched_det.x1, matched_det.y1, matched_det.x2, matched_det.y2)
                track.confidence = matched_det.confidence
                track.visible = True
                track.last_seen = timestamp
                track.state = TrackLifecycle.VISIBLE
                matched_track_ids.add(track_id)

        # Step 2: Mark unmatched tracks as TEMPORARILY_LOST or remove if expired
        expired_track_ids = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                if (timestamp - track.last_seen) > self.lost_timeout:
                    track.state = TrackLifecycle.REMOVED
                    expired_track_ids.append(track_id)
                else:
                    track.visible = False
                    track.state = TrackLifecycle.TEMPORARILY_LOST

        for track_id in expired_track_ids:
            del self._tracks[track_id]

        # Step 3: Create new tracks for remaining unmatched detections
        for det in unmatched_detections:
            pos = Vector2(det.center[0], det.center[1])
            new_track = Track(
                track_id=self._next_track_id,
                position=pos,
                bbox=(det.x1, det.y1, det.x2, det.y2),
                confidence=det.confidence,
                class_name=det.class_name,
                class_id=det.class_id,
                visible=True,
                first_seen=timestamp,
                last_seen=timestamp,
                state=TrackLifecycle.NEW,
            )
            self._tracks[self._next_track_id] = new_track
            self._next_track_id += 1

        return list(self._tracks.values())

    def get_active_tracks(self) -> List[Track]:
        """Returns all non-expired tracks."""
        return list(self._tracks.values())

    def get_visible_tracks(self) -> List[Track]:
        """Returns only tracks that are visible in the most recent update."""
        return [t for t in self._tracks.values() if t.visible]

    def reset(self) -> None:
        """Clears all tracks and resets ID sequence."""
        self._tracks.clear()
        self._next_track_id = 1
