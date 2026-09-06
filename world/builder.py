"""
world.builder — Canonical WorldState builder for Claude Tactical AI V2.

Translates computer vision detections, tracking lifecycle data, and optical sensor
readings into an immutable WorldState snapshot.
"""

from typing import List, Optional, Tuple
from world.models import (
    WorldState,
    PlayerState,
    EnemyState,
    MinionState,
    TurretState,
    Vector2,
)
from vision.detector import Detection
from vision.hp_detector import HPObservation
from vision.skill_state import SkillState
from world.tracker import Track


class WorldStateBuilder:
    """
    Constructs a unified WorldState contract from detections, tracks, and sensors.
    """

    DEFAULT_HERO_POS = Vector2(772.0, 360.0)

    def build(
        self,
        timestamp: float,
        detections: Optional[List[Detection]] = None,
        tracks: Optional[List[Track]] = None,
        player: Optional[PlayerState] = None,
        player_hp: Optional[HPObservation] = None,
        skills: Optional[SkillState] = None,
        is_near_base: bool = False,
    ) -> WorldState:
        """
        Builds a WorldState snapshot.

        Args:
            timestamp: Monotonic timestamp for this observation frame.
            detections: Raw YOLO detections on canonical 1544x720 canvas.
            tracks: Active tracked entities from WorldTracker.
            player: Explicit PlayerState if already known.
            player_hp: Optical HP reading for player hero.
            skills: Optical skill availability state.
            is_near_base: Whether our hero is near the fountain base.

        Returns:
            WorldState dataclass instance.
        """
        detections = detections or []
        enemies: List[EnemyState] = []
        minions: List[MinionState] = []
        turrets: List[TurretState] = []

        detected_player_pos: Optional[Vector2] = None
        detected_player_visible: bool = False

        for det in detections:
            c_name = det.class_name.lower()
            cx, cy = det.center
            bbox = (det.x1, det.y1, det.x2, det.y2)
            pos = Vector2(cx, cy)

            if any(k in c_name for k in ("enemy", "hero_enemy", "hp_enemy")):
                enemies.append(
                    EnemyState(
                        position=pos,
                        bbox=bbox,
                        confidence=det.confidence,
                        source="yolo",
                    )
                )
            elif any(k in c_name for k in ("minion", "creep", "buff", "monster", "crab")):
                minions.append(
                    MinionState(
                        position=pos,
                        bbox=bbox,
                        confidence=det.confidence,
                        is_enemy=True,
                        kind="minion" if "minion" in c_name else "buff",
                    )
                )
            elif any(k in c_name for k in ("turret", "tower")):
                turrets.append(
                    TurretState(
                        position=pos,
                        bbox=bbox,
                        confidence=det.confidence,
                        is_enemy=True,
                    )
                )
            elif any(k in c_name for k in ("player", "self", "hp_self", "hero_self")):
                detected_player_pos = pos
                detected_player_visible = True

        # Resolve PlayerState: explicit parameter -> detected player -> default position
        if player is None:
            if detected_player_pos is not None:
                player = PlayerState(
                    position=detected_player_pos,
                    hp=player_hp or HPObservation(value=1.0, confidence=1.0, visible=True),
                    is_visible=True,
                    is_dead=False,
                )
            elif player_hp is not None and player_hp.visible:
                player = PlayerState(
                    position=self.DEFAULT_HERO_POS,
                    hp=player_hp,
                    is_visible=True,
                    is_dead=False,
                )
            else:
                # LOST != DEAD: hero temporarily not detected in frame
                player = PlayerState(
                    position=self.DEFAULT_HERO_POS,
                    hp=player_hp or HPObservation(value=1.0, confidence=0.0, visible=False),
                    is_visible=False,
                    is_dead=False,
                )

        return WorldState(
            timestamp=float(timestamp),
            player=player,
            enemies=enemies,
            minions=minions,
            turrets=turrets,
            skills=skills,
            is_near_base=bool(is_near_base),
        )
