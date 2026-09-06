"""Regression scenarios covering all 14 V1 Brain decisions.

Defines the BrainScenario data model, static regression scenarios,
and the canonical V1 decision string to V2 ActionType mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from actions.models import ActionType
from brain.state import TacticalState
from world.models import Vector2


@dataclass(frozen=True)
class BrainScenario:
    """An immutable regression test scenario comparing V1 and V2 decision output."""

    name: str
    state: TacticalState
    expected_v1_decision: str


# Canonical 1:1 mapping from V1 decision strings to V2 typed ActionType
V1_TO_V2_ACTION_MAP: Dict[str, ActionType] = {
    "TURRET_RETREAT": ActionType.RETREAT,
    "TACTICAL_RETREAT": ActionType.RETREAT,
    "KITE_AND_POKE": ActionType.KITE,
    "SWEET_SPOT_BURST": ActionType.ATTACK,
    "DIVE_ALL_IN": ActionType.CAST_ULT,
    "FARM_APPROACH": ActionType.MOVE,
    "FARM_KITE_BACK": ActionType.KITE,
    "FARM_SWEET_SPOT": ActionType.FARM,
    "FARM_S1_AOE": ActionType.CAST_S1,
    "LANE_ADVANCE": ActionType.MOVE,
    "LANE_PUSH": ActionType.MOVE,
    "RIVER_SCOUT": ActionType.SCOUT,
    "FLANK_ADVANCE": ActionType.SCOUT,
    "CREEP_INTERCEPT": ActionType.FARM,
}


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
    enemy_dist: Optional[float] = None,
    enemy_pos: Optional[Vector2] = None,
    enemy_id: Optional[int] = None,
    minion_count: int = 0,
    minion_dist: Optional[float] = None,
    minion_pos: Optional[Vector2] = None,
    minion_id: Optional[int] = None,
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


def get_all_scenarios() -> List[BrainScenario]:
    """Builds a deterministic suite of all 14 V1 decision scenarios."""
    player_default = Vector2(772.0, 360.0)
    enemy_default = Vector2(900.0, 360.0)
    minion_default = Vector2(850.0, 360.0)

    scenarios = [
        # --- Combat Decisions ---
        BrainScenario(
            name="scenario_turret_retreat",
            expected_v1_decision="TURRET_RETREAT",
            state=_make_state(
                enemy_visible=True,
                enemy_count=1,
                enemy_dist=240.0,
                enemy_pos=enemy_default,
                enemy_id=10,
                turret_near=True,
                hp_ratio=0.90,
            ),
        ),
        BrainScenario(
            name="scenario_tactical_retreat",
            expected_v1_decision="TACTICAL_RETREAT",
            state=_make_state(
                enemy_visible=True,
                enemy_count=1,
                enemy_dist=240.0,
                enemy_pos=enemy_default,
                enemy_id=11,
                turret_near=False,
                hp_ratio=0.20,  # < 0.32 triggers tactical retreat
            ),
        ),
        BrainScenario(
            name="scenario_kite_and_poke",
            expected_v1_decision="KITE_AND_POKE",
            state=_make_state(
                enemy_visible=True,
                enemy_count=1,
                enemy_dist=200.0,  # DANGER_CLOSE (< 260)
                enemy_pos=enemy_default,
                enemy_id=12,
                turret_near=False,
                hp_ratio=0.80,
                can_combo=False,
            ),
        ),
        BrainScenario(
            name="scenario_sweet_spot_burst",
            expected_v1_decision="SWEET_SPOT_BURST",
            state=_make_state(
                enemy_visible=True,
                enemy_count=1,
                enemy_dist=350.0,  # SWEET_SPOT (260..430)
                enemy_pos=enemy_default,
                enemy_id=13,
                turret_near=False,
                hp_ratio=0.80,
                can_combo=False,
            ),
        ),
        BrainScenario(
            name="scenario_dive_all_in",
            expected_v1_decision="DIVE_ALL_IN",
            state=_make_state(
                enemy_visible=True,
                enemy_count=1,
                enemy_dist=350.0,  # SWEET_SPOT
                enemy_pos=enemy_default,
                enemy_id=14,
                turret_near=False,
                hp_ratio=0.80,
                can_combo=True,
            ),
        ),
        # --- Farm Decisions ---
        BrainScenario(
            name="scenario_farm_approach",
            expected_v1_decision="FARM_APPROACH",
            state=_make_state(
                enemy_visible=False,
                minion_count=2,
                minion_dist=500.0,  # CREEP_FAR (> 420)
                minion_pos=minion_default,
                minion_id=20,
                hp_ratio=0.80,
                s1_ready=True,
            ),
        ),
        BrainScenario(
            name="scenario_farm_kite_back",
            expected_v1_decision="FARM_KITE_BACK",
            state=_make_state(
                enemy_visible=False,
                minion_count=2,
                minion_dist=200.0,  # CREEP_DANGER_CLOSE (< 260)
                minion_pos=minion_default,
                minion_id=21,
                hp_ratio=0.80,
                s1_ready=False,
            ),
        ),
        BrainScenario(
            name="scenario_farm_sweet_spot",
            expected_v1_decision="FARM_SWEET_SPOT",
            state=_make_state(
                enemy_visible=False,
                minion_count=2,
                minion_dist=320.0,  # CREEP_SWEET_SPOT (260..420)
                minion_pos=minion_default,
                minion_id=22,
                hp_ratio=0.80,
                s1_ready=False,
            ),
        ),
        BrainScenario(
            name="scenario_farm_s1_aoe",
            expected_v1_decision="FARM_S1_AOE",
            state=_make_state(
                enemy_visible=False,
                minion_count=2,
                minion_dist=320.0,  # CREEP_SWEET_SPOT
                minion_pos=minion_default,
                minion_id=23,
                hp_ratio=0.80,
                s1_ready=True,
            ),
        ),
        # --- Roam Decisions ---
        BrainScenario(
            name="scenario_lane_advance",
            expected_v1_decision="LANE_ADVANCE",
            state=_make_state(
                timestamp=2.0,  # SCOUT_HOT_ZONE (< 5.0)
                enemy_visible=False,
                minion_count=0,
            ),
        ),
        BrainScenario(
            name="scenario_lane_push",
            expected_v1_decision="LANE_PUSH",
            state=_make_state(
                timestamp=8.0,  # SCOUT_MID_SEARCH (5.0..15.0)
                enemy_visible=False,
                minion_count=0,
            ),
        ),
        BrainScenario(
            name="scenario_river_scout",
            expected_v1_decision="RIVER_SCOUT",
            state=_make_state(
                timestamp=25.0,  # SCOUT_DEEP_PATROL (>= 15.0)
                enemy_visible=False,
                minion_count=0,
            ),
        ),
        BrainScenario(
            name="scenario_flank_advance",
            expected_v1_decision="FLANK_ADVANCE",
            state=_make_state(
                timestamp=30.0,
                enemy_visible=False,
                minion_count=0,
            ),
        ),
        BrainScenario(
            name="scenario_creep_intercept",
            expected_v1_decision="CREEP_INTERCEPT",
            state=_make_state(
                timestamp=35.0,
                enemy_visible=False,
                minion_count=0,
            ),
        ),
    ]

    return scenarios


def create_scenario_q_tables() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Generates isolated Q-tables ensuring deterministic exploitation of scenario target actions."""
    combat_q = {
        "DANGER_CLOSE_HIGH_COMBO_False": {
            "KITE_AND_POKE": 100.0,
            "SWEET_SPOT_BURST": 10.0,
            "DIVE_ALL_IN": 0.0,
        },
        "SWEET_SPOT_HIGH_COMBO_False": {
            "KITE_AND_POKE": 10.0,
            "SWEET_SPOT_BURST": 100.0,
            "DIVE_ALL_IN": 0.0,
        },
        "SWEET_SPOT_HIGH_COMBO_True": {
            "KITE_AND_POKE": 10.0,
            "SWEET_SPOT_BURST": 20.0,
            "DIVE_ALL_IN": 100.0,
        },
    }

    farm_q = {
        "CREEP_FAR_S1_True_SAFE": {
            "FARM_APPROACH": 100.0,
            "FARM_KITE_BACK": 0.0,
            "FARM_SWEET_SPOT": 0.0,
            "FARM_S1_AOE": 0.0,
        },
        "CREEP_DANGER_CLOSE_S1_False_SAFE": {
            "FARM_APPROACH": 0.0,
            "FARM_KITE_BACK": 100.0,
            "FARM_SWEET_SPOT": 0.0,
            "FARM_S1_AOE": 0.0,
        },
        "CREEP_SWEET_SPOT_S1_False_SAFE": {
            "FARM_APPROACH": 0.0,
            "FARM_KITE_BACK": 0.0,
            "FARM_SWEET_SPOT": 100.0,
            "FARM_S1_AOE": 0.0,
        },
        "CREEP_SWEET_SPOT_S1_True_SAFE": {
            "FARM_APPROACH": 0.0,
            "FARM_KITE_BACK": 0.0,
            "FARM_SWEET_SPOT": 10.0,
            "FARM_S1_AOE": 100.0,
        },
    }

    # Roam Q-table: target each scenario's roam action
    roam_q = {
        "SCOUT_HOT_ZONE": {
            "LANE_ADVANCE": 100.0,
            "LANE_PUSH": 10.0,
            "RIVER_SCOUT": 10.0,
            "FLANK_ADVANCE": 10.0,
            "CREEP_INTERCEPT": 10.0,
        },
        "SCOUT_MID_SEARCH": {
            "LANE_ADVANCE": 10.0,
            "LANE_PUSH": 100.0,
            "RIVER_SCOUT": 10.0,
            "FLANK_ADVANCE": 10.0,
            "CREEP_INTERCEPT": 10.0,
        },
        "SCOUT_DEEP_PATROL": {
            "LANE_ADVANCE": 10.0,
            "LANE_PUSH": 10.0,
            "RIVER_SCOUT": 100.0,
            "FLANK_ADVANCE": 10.0,
            "CREEP_INTERCEPT": 10.0,
        },
    }

    return roam_q, farm_q, combat_q
