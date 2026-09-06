"""
brain.tactical — High-level tactical decision engine contract for Claude Tactical AI V2.

Contract:
    WorldState → TacticalBrain.decide() → Action

Pure cognitive layer:
- Interprets observed battlefield state (WorldState).
- Chooses appropriate abstract tactical Action.
- Zero awareness of ADB, Android touchscreen coordinates, joystick driver,
  or optical frame processing (OpenCV/YOLO).
"""

from typing import Optional
from world.models import WorldState
from actions.models import Action, ActionType


class TacticalBrain:
    """
    Tactical Decision Engine contract.
    Evaluates battlefield WorldState and produces high-level Actions.
    """

    def __init__(self):
        # Prepared structure for safety rules, state history, and tactical policies
        pass

    def decide(self, world_state: WorldState) -> Action:
        """
        Evaluates current WorldState and returns next abstract Action.

        Args:
            world_state: Snapshot of player, enemies, creeps, turrets, and skills.

        Returns:
            Action: Concrete intent (MOVE, ATTACK, KITE, RETREAT, etc.)
        """
        if not isinstance(world_state, WorldState):
            raise TypeError(f"TacticalBrain.decide expects WorldState, got: {type(world_state)}")

        # Baseline deterministic contract: returns IDLE action
        return Action(type=ActionType.IDLE)
