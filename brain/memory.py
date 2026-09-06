"""Brain memory and persistence layer for Claude Tactical AI V2.

Isolates disk persistence (loading and saving Q-tables, highlight moments, decision counters)
and experience tracking from V1 claude_brain.py without performing Q-table mutations.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ActionRecord:
    """A recorded action step in the sliding history buffer."""

    category: str
    state: str
    action: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "state": self.state,
            "action": self.action,
            "time": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActionRecord:
        return cls(
            category=str(data.get("category", "")),
            state=str(data.get("state", "")),
            action=str(data.get("action", "")),
            timestamp=float(data.get("time", 0.0)),
        )


@dataclass(frozen=True)
class Experience:
    """A triumph / highlight moment recorded in the brain memory ("Golden Fund")."""

    id: int
    type: str
    reward: float
    hp_pct: int
    actions: Tuple[str, ...]
    description: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "reward": self.reward,
            "hp_pct": self.hp_pct,
            "actions": list(self.actions),
            "description": self.description,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Experience:
        actions_raw = data.get("actions", [])
        if isinstance(actions_raw, (list, tuple)):
            actions_tuple = tuple(str(a) for a in actions_raw)
        else:
            actions_tuple = ()

        return cls(
            id=int(data.get("id", 0)),
            type=str(data.get("type", "")),
            reward=float(data.get("reward", 0.0)),
            hp_pct=int(data.get("hp_pct", 100)),
            actions=actions_tuple,
            description=str(data.get("description", "")),
            timestamp=str(data.get("timestamp", "")),
        )


class BrainMemory:
    """Handles storage, serialization, and retrieval of brain experiences and Q-tables.

    Strictly responsible for data storage and persistence. Does NOT mutate Q-values
    or execute reinforcement learning updates.
    """

    MAX_MOMENTS_TO_PERSIST: int = 50
    ACTION_HISTORY_LIMIT: int = 30

    def __init__(self, memory_file: Optional[str] = "q_brain.json") -> None:
        self.memory_file: Optional[str] = memory_file

        # Q-tables (state -> {action: weight})
        self.roam_q_table: Dict[str, Dict[str, float]] = {}
        self.farm_q_table: Dict[str, Dict[str, float]] = {}
        self.combat_q_table: Dict[str, Dict[str, float]] = {}

        # Highlight memory bank
        self.best_moments: List[Experience] = []

        # Sliding action history
        self.recent_action_history: List[ActionRecord] = []

        self.total_decisions: int = 0
        self.saved_at: Optional[str] = None
        self.last_highlight_text: str = ""

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------
    def load(self, path: Optional[str] = None) -> bool:
        """Loads brain memory from JSON file.

        Compatible 1:1 with V1 q_brain.json schema.
        Returns True if loaded, False if file doesn't exist or on read error.
        """
        target_path = path or self.memory_file
        if not target_path or not os.path.exists(target_path):
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.roam_q_table = data.get("roam_q", {})
            self.farm_q_table = data.get("farm_q", {})
            self.combat_q_table = data.get("combat_q", {})

            raw_moments = data.get("best_moments", [])
            self.best_moments = [
                Experience.from_dict(m) if isinstance(m, dict) else m
                for m in raw_moments
            ]

            self.total_decisions = int(data.get("total_decisions", 0))
            self.saved_at = data.get("saved_at")
            return True
        except Exception:
            return False

    def save(self, path: Optional[str] = None, timestamp_str: Optional[str] = None) -> bool:
        """Saves brain memory to disk using exact V1 schema.

        Retains the last 50 best moments as in V1.
        Returns True on success, False on error.
        """
        target_path = path or self.memory_file
        if not target_path:
            return False

        ts = timestamp_str or time.strftime("%Y-%m-%d %H:%M:%S")
        self.saved_at = ts

        # Only persist the most recent 50 moments (exact V1 behavior)
        moments_to_save = [
            m.to_dict() if isinstance(m, Experience) else m
            for m in self.best_moments[-self.MAX_MOMENTS_TO_PERSIST:]
        ]

        payload = {
            "roam_q": self.roam_q_table,
            "farm_q": self.farm_q_table,
            "combat_q": self.combat_q_table,
            "best_moments": moments_to_save,
            "total_decisions": self.total_decisions,
            "saved_at": ts,
        }

        try:
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    # ----------------------------------------------------------------------
    # State inspection and replacement
    # ----------------------------------------------------------------------
    def get_data(self) -> Dict[str, Any]:
        """Returns full internal state dictionary for serialization or inspection."""
        return {
            "roam_q": copy.deepcopy(self.roam_q_table),
            "farm_q": copy.deepcopy(self.farm_q_table),
            "combat_q": copy.deepcopy(self.combat_q_table),
            "best_moments": [m.to_dict() for m in self.best_moments],
            "total_decisions": self.total_decisions,
            "saved_at": self.saved_at,
        }

    def replace_data(self, data: Dict[str, Any]) -> None:
        """Replaces memory content from a dictionary."""
        self.roam_q_table = copy.deepcopy(data.get("roam_q", {}))
        self.farm_q_table = copy.deepcopy(data.get("farm_q", {}))
        self.combat_q_table = copy.deepcopy(data.get("combat_q", {}))

        raw_moments = data.get("best_moments", [])
        self.best_moments = [
            Experience.from_dict(m) if isinstance(m, dict) else m
            for m in raw_moments
        ]
        self.total_decisions = int(data.get("total_decisions", 0))
        self.saved_at = data.get("saved_at")

    def reset(self) -> None:
        """Clears all tables, moments, action history, and counters."""
        self.roam_q_table.clear()
        self.farm_q_table.clear()
        self.combat_q_table.clear()
        self.best_moments.clear()
        self.recent_action_history.clear()
        self.total_decisions = 0
        self.saved_at = None
        self.last_highlight_text = ""

    # ----------------------------------------------------------------------
    # Action history and Highlight storage
    # ----------------------------------------------------------------------
    def log_action(
        self,
        category: str,
        state: str,
        action: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Logs an action to the sliding action history (maximum 30 elements)."""
        ts = timestamp if timestamp is not None else time.time()
        record = ActionRecord(
            category=category,
            state=state,
            action=action,
            timestamp=ts,
        )
        self.recent_action_history.append(record)
        if len(self.recent_action_history) > self.ACTION_HISTORY_LIMIT:
            self.recent_action_history = self.recent_action_history[-self.ACTION_HISTORY_LIMIT:]

    def get_recent_actions(
        self,
        since_seconds: float = 8.0,
        limit: int = 6,
        current_time: Optional[float] = None,
    ) -> List[ActionRecord]:
        """Returns the most recent actions within the time window."""
        now = current_time if current_time is not None else time.time()
        recent = [a for a in self.recent_action_history if (now - a.timestamp) <= since_seconds]
        return recent[-limit:]

    def record_moment(self, moment: Experience) -> None:
        """Adds an experience/highlight moment to memory bank."""
        self.best_moments.append(moment)
        self.last_highlight_text = f"[{moment.type}] {moment.description} (+{int(moment.reward)} 🥕)"

    def create_moment_from_history(
        self,
        moment_type: str,
        reward: float,
        description: str,
        hp_left_pct: int = 100,
        current_time: Optional[float] = None,
        timestamp_str: Optional[str] = None,
    ) -> Tuple[Experience, List[ActionRecord]]:
        """Constructs and stores a new Experience moment linked with recent actions.

        Returns (new_moment, recent_actions_chain).
        Does NOT modify Q-tables (responsibility belongs to Q-learning replay).
        """
        now = current_time if current_time is not None else time.time()
        recent = self.get_recent_actions(since_seconds=8.0, limit=6, current_time=now)
        action_names = tuple(a.action for a in recent)

        ts = timestamp_str or time.strftime("%Y-%m-%d %H:%M:%S")
        moment = Experience(
            id=len(self.best_moments) + 1,
            type=moment_type,
            reward=reward,
            hp_pct=hp_left_pct,
            actions=action_names,
            description=description,
            timestamp=ts,
        )
        self.record_moment(moment)
        return moment, recent

    def recall_best_tactic(
        self,
        allowed_actions: Sequence[str],
        window_size: int = 10,
    ) -> Optional[str]:
        """Recalls the first matching winning tactic from recent best moments."""
        if not self.best_moments:
            return None

        allowed_set = set(allowed_actions)
        for m in reversed(self.best_moments[-window_size:]):
            for act in m.actions:
                if act in allowed_set:
                    return act
        return None

    # ----------------------------------------------------------------------
    # Interoperability with QLearningCore
    # ----------------------------------------------------------------------
    def export_q_tables(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Returns Q-tables dictionary compatible with QLearningCore.load_tables."""
        return {
            "roam_q": copy.deepcopy(self.roam_q_table),
            "farm_q": copy.deepcopy(self.farm_q_table),
            "combat_q": copy.deepcopy(self.combat_q_table),
        }

    def import_q_tables(self, tables: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Imports Q-tables from QLearningCore.export_tables."""
        self.roam_q_table = copy.deepcopy(tables.get("roam_q", {}))
        self.farm_q_table = copy.deepcopy(tables.get("farm_q", {}))
        self.combat_q_table = copy.deepcopy(tables.get("combat_q", {}))
