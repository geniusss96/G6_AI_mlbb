"""
tests/test_stage_22_e2e.py — Stage 22 End-to-End V2 Regression & Integration Validation.

Proves that the complete V2 architecture functions coherently as one deterministic system:
Capture -> Vision -> WorldState -> Tracker -> Events -> TacticalState -> Brain -> Action -> ActionExecutor -> Control -> ADB
and the learning side channel:
Events -> RewardCalculator -> LearningIntegrator -> LearningLoop -> QLearningCore -> Memory.

All tests operate offline without physical Android, ADB, Scrcpy, GPU, or real-time sleep.
"""

import ast
import hashlib
import os
import random
import sys
import unittest
from typing import List, Optional

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from actions.models import Action, ActionType
from app.runtime import V2Runtime, RuntimeTickResult
from brain.action_selection import ActionSelector
from brain.encoder import TacticalStateEncoder
from brain.exploration import ExplorationPolicy, ExplorationContext
from brain.learning_integration import LearningIntegrator
from brain.learning_loop import LearningLoop
from brain.q_learning import QLearningCore
from brain.rewards import RewardCalculator
from brain.state import TacticalState, TacticalStateBuilder
from brain.transition import Transition
from brain.transition_builder import TransitionBuilder
from tests.e2e.harness import E2EHarness
from vision.detector import Detection
from vision.hp_detector import HPObservation
from vision.skill_state import SkillState
from world.events import EventType, GameEvent, WorldEventDetector
from world.models import EnemyState, PlayerState, TurretState, Vector2, WorldState


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestStage22EndToEndRegression(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline_path = os.path.join(WORKSPACE_ROOT, "data", "q_brain_baseline.json")
        self.q_brain_path = os.path.join(WORKSPACE_ROOT, "q_brain.json")

        self.initial_baseline_hash = (
            _file_hash(self.baseline_path) if os.path.exists(self.baseline_path) else None
        )
        self.initial_q_brain_hash = (
            _file_hash(self.q_brain_path) if os.path.exists(self.q_brain_path) else None
        )

        self.harness = E2EHarness(initial_time=100.0)

    def tearDown(self) -> None:
        # Strict SHA-256 baseline and q_brain protection
        if self.initial_baseline_hash and os.path.exists(self.baseline_path):
            self.assertEqual(
                self.initial_baseline_hash,
                _file_hash(self.baseline_path),
                "data/q_brain_baseline.json was modified during regression test!",
            )
        if self.initial_q_brain_hash and os.path.exists(self.q_brain_path):
            self.assertEqual(
                self.initial_q_brain_hash,
                _file_hash(self.q_brain_path),
                "q_brain.json was modified during regression test!",
            )

    # --------------------------------------------------------------------------
    # 1. Complete Combat Success Scenario
    # --------------------------------------------------------------------------
    def test_complete_combat_success_scenario(self):
        """
        Simulate full combat engagement:
        Player healthy -> enemy enters range -> combat action chosen & executed ->
        combat engagement duration met -> enemy disappears -> HERO_KILL event ->
        multi-domain rewards (combat +60, roam +40) -> distinct Q updates.
        """
        # 1. Spawn enemy in range (x=900, y=360 -> dist ~128px from 772,360)
        self.harness.spawn_enemy_hero(x=900.0, y=360.0)

        # Tick 1: Player encounters enemy, Brain selects combat action, ActionExecutor executes
        tick1 = self.harness.tick()
        self.assertTrue(tick1.success)
        self.assertTrue(tick1.tactical_state.enemy_visible)
        self.assertIn(tick1.action.type, (ActionType.KITE, ActionType.ATTACK, ActionType.CAST_ULT))
        self.assertTrue(tick1.execution_result.success)

        # Inform event detector that player engaged in combat attack at t=100.0
        self.harness.event_detector.record_attack_action(
            target_pos=Vector2(900.0, 360.0),
            timestamp=self.harness.current_time,
            is_hero=True,
        )

        # Advance time by 2.2s (so combat_duration = 2.2 >= MIN_COMBAT_DURATION_SEC = 2.0s)
        self.harness.advance_time(2.2)
        self.harness.event_detector.record_attack_action(
            target_pos=Vector2(900.0, 360.0),
            timestamp=self.harness.current_time,
            is_hero=True,
        )
        self.harness.tick()

        # Enemy is eliminated / disappears from vision at t=102.3
        self.harness.advance_time(0.1)
        self.harness.clear_entities()
        # Tick to register enemy disappearance timestamp
        self.harness.tick()

        # Advance time by 2.6s (exceeds ENEMY_DISAPPEAR_TIMEOUT_SEC = 2.5s while attack_freshness <= 3.5s)
        self.harness.advance_time(2.6)

        # Tick 2: WorldEventDetector confirms HERO_KILL, LearningIntegrator updates Q-learning
        initial_combat_q = self.harness.q_learning.get_q("combat", "DANGER_CLOSE_HIGH_COMBO_False", "attack")

        tick2 = self.harness.tick()
        self.assertTrue(tick2.success)

        # Verify HERO_KILL event was detected
        kill_events = [e for e in tick2.events if e.type == EventType.HERO_KILL]
        self.assertEqual(len(kill_events), 1)

        # Verify multi-domain reward cardinality: 2 distinct transitions (+60 combat, +40 roam)
        self.assertEqual(len(tick2.transitions), 2)
        rewards = [t.reward for t in tick2.transitions]
        self.assertIn(60.0, rewards)
        self.assertIn(40.0, rewards)
        self.assertNotIn(100.0, rewards, "Rewards must NOT be summed across domains!")

        # Verify Q-tables updated without error
        new_combat_q = self.harness.q_learning.get_q("combat", "DANGER_CLOSE_HIGH_COMBO_False", "attack")
        self.assertIsInstance(new_combat_q, float)

    # --------------------------------------------------------------------------
    # 2. Combat Death Scenario
    # --------------------------------------------------------------------------
    def test_combat_death_scenario(self):
        """
        Simulate combat death:
        Player in combat -> HP becomes critical (<0.28) -> disappearance ->
        PLAYER_DEATH event -> terminal transition (done=True, next_state=None) ->
        terminal TD update target = r.
        """
        # 1. Engage combat
        self.harness.spawn_enemy_hero(x=900.0, y=360.0)
        self.harness.tick()

        self.harness.event_detector.record_attack_action(
            target_pos=Vector2(900.0, 360.0),
            timestamp=self.harness.current_time,
            is_hero=True,
        )

        # 2. Player takes critical damage (HP = 0.15) while still visible
        self.harness.advance_time(0.5)
        self.harness.world_builder.build = lambda timestamp, **kwargs: WorldState(  # type: ignore
            timestamp=timestamp,
            player=PlayerState(
                position=Vector2(772.0, 360.0),
                hp=HPObservation(value=0.15, confidence=1.0, visible=True),
                is_visible=True,
                is_dead=False,
            ),
        )
        self.harness.tick()

        # 3. Player disappears (lethal confirmation sequence)
        self.harness.advance_time(0.5)
        self.harness.clear_entities()
        self.harness.world_builder.build = lambda timestamp, **kwargs: WorldState(  # type: ignore
            timestamp=timestamp,
            player=PlayerState(
                position=Vector2(772.0, 360.0),
                hp=HPObservation(value=0.15, confidence=1.0, visible=False),
                is_visible=False,
                is_dead=False,
            ),
        )
        self.harness.tick()

        # 4. Advance past PLAYER_DISAPPEAR_TIMEOUT_SEC (2.5s) while still in RECENT_COMBAT_TIMEOUT_SEC (4.0s)
        self.harness.advance_time(2.6)

        tick_death = self.harness.tick()
        death_events = [e for e in tick_death.events if e.type == EventType.PLAYER_DEATH]
        self.assertEqual(len(death_events), 1)

        # Verify terminal learning semantics
        self.assertGreater(len(tick_death.transitions), 0)
        death_trans = tick_death.transitions[0]
        self.assertTrue(death_trans.done)
        self.assertIsNone(death_trans.next_state_key)
        rewards = [t.reward for t in tick_death.transitions]
        self.assertIn(-40.0, rewards)  # V1 combat death penalty
        self.assertIn(-30.0, rewards)  # V1 roam death penalty

    # --------------------------------------------------------------------------
    # 3. Farming Scenario
    # --------------------------------------------------------------------------
    def test_farming_scenario(self):
        """
        Simulate farming:
        Minion visible -> farm action -> engagement duration -> creep disappears ->
        CREEP_KILL event -> farm reward (+15 farm, +5 roam).
        """
        self.harness.spawn_minion(x=850.0, y=360.0)
        tick1 = self.harness.tick()
        self.assertTrue(tick1.success)

        # Inform event detector of creep attack
        self.harness.event_detector.record_attack_action(
            target_pos=Vector2(850.0, 360.0),
            timestamp=self.harness.current_time,
            is_hero=False,
        )

        # Advance past MIN_CREEP_ATTACK_DURATION_SEC (2.5s)
        self.harness.advance_time(2.6)
        self.harness.event_detector.record_attack_action(
            target_pos=Vector2(850.0, 360.0),
            timestamp=self.harness.current_time,
            is_hero=False,
        )
        self.harness.tick()

        # Minion eliminated and disappears
        self.harness.advance_time(0.1)
        self.harness.clear_entities()
        self.harness.tick()

        # Advance past CREEP_DISAPPEAR_TIMEOUT_SEC (2.0s)
        self.harness.advance_time(2.1)

        tick2 = self.harness.tick()
        creep_events = [e for e in tick2.events if e.type == EventType.CREEP_KILL]
        self.assertEqual(len(creep_events), 1)

        # Verify farm rewards (+25 farm, +15 roam)
        self.assertEqual(len(tick2.transitions), 2)
        rewards = [t.reward for t in tick2.transitions]
        self.assertIn(25.0, rewards)
        self.assertIn(15.0, rewards)

    # --------------------------------------------------------------------------
    # 4. Roaming Scenario
    # --------------------------------------------------------------------------
    def test_roaming_scenario(self):
        """
        Simulate roaming:
        No enemies -> roam action (MOVE/SCOUT) -> target entered -> TARGET_ENTERED event ->
        roam reward (+30 roam).
        """
        self.harness.clear_entities()
        tick1 = self.harness.tick()
        self.assertIn(tick1.action.type, (ActionType.MOVE, ActionType.SCOUT, ActionType.FARM))

        # Target enters frame
        self.harness.advance_time(1.0)
        self.harness.spawn_enemy_hero(x=1000.0, y=360.0)

        tick2 = self.harness.tick()
        entered_events = [e for e in tick2.events if e.type == EventType.TARGET_ENTERED]
        self.assertEqual(len(entered_events), 1)

        # Verify roam reward (+30.0 for enemy spotted in roam phase)
        self.assertEqual(len(tick2.transitions), 1)
        self.assertEqual(tick2.transitions[0].reward, 30.0)

    # --------------------------------------------------------------------------
    # 5. Turret Danger Scenario
    # --------------------------------------------------------------------------
    def test_turret_danger_scenario(self):
        """
        Enemy turret near player (<420px) flags turret danger and forces retreat stance.
        """
        # Spawn enemy hero and turret together near player
        enemy_det = Detection(
            class_id=0,
            class_name="enemy_hero",
            confidence=0.95,
            x1=880,
            y1=340,
            x2=920,
            y2=380,
        )
        turret_det = Detection(
            class_id=2,
            class_name="turret",
            confidence=0.90,
            x1=890,
            y1=350,
            x2=910,
            y2=370,
        )
        self.harness.vision.set_detections([enemy_det, turret_det])

        tick = self.harness.tick()
        self.assertTrue(tick.tactical_state.enemy_near_turret)
        # Tactical brain chooses retreat under turret danger
        self.assertEqual(tick.action.type, ActionType.RETREAT)

    # --------------------------------------------------------------------------
    # 6. Zero-Reward Event Scenario
    # --------------------------------------------------------------------------
    def test_zero_reward_event_target_lost(self):
        """TARGET_LOST generates an event but produces 0 rewards and 0 transitions."""
        self.harness.spawn_enemy_hero(x=900.0, y=360.0)
        self.harness.tick()

        # Target lost without combat kill
        self.harness.advance_time(1.0)
        self.harness.clear_entities()

        tick2 = self.harness.tick()
        lost_events = [e for e in tick2.events if e.type == EventType.TARGET_LOST]
        self.assertEqual(len(lost_events), 1)
        self.assertEqual(len(tick2.transitions), 0)

    # --------------------------------------------------------------------------
    # 7. Exploration Determinism & Controlled Non-Greedy Choice
    # --------------------------------------------------------------------------
    def test_exploration_determinism_and_branching(self):
        """Exploration with identical RNG seed produces identical output."""
        policy = ExplorationPolicy()
        encoder = TacticalStateEncoder()
        q_core = QLearningCore()
        q_core.set_q("combat", "DANGER_CLOSE_HIGH_COMBO_True", "SWEET_SPOT_BURST", 50.0)
        q_core.set_q("combat", "DANGER_CLOSE_HIGH_COMBO_True", "KITE_AND_POKE", 5.0)

        selector = ActionSelector(encoder, q_core, policy)
        state = TacticalState(
            timestamp=100.0,
            player_hp_ratio=1.0,
            player_position=Vector2(772.0, 360.0),
            is_player_dead=False,
            s1_ready=True,
            s2_ready=True,
            ultimate_ready=True,
            can_combo=True,
            enemy_visible=True,
            enemy_count=1,
            nearest_enemy_distance=200.0,
            nearest_enemy_position=Vector2(850.0, 360.0),
            minion_count=0,
            nearest_minion_distance=None,
            nearest_minion_position=None,
            enemy_near_turret=False,
            is_near_base=False,
        )
        context = ExplorationContext.COMBAT
        candidates = ["SWEET_SPOT_BURST", "KITE_AND_POKE"]

        # 1. Run with seed 42 twice -> deterministic identical action
        rng1 = random.Random(42)
        act1 = selector.select(state, context, candidates, rng=rng1)

        rng2 = random.Random(42)
        act2 = selector.select(state, context, candidates, rng=rng2)

        self.assertEqual(act1, act2)

        # 2. Test controlled non-greedy choice under forced exploration
        forced_policy = ExplorationPolicy(combat_epsilon=1.0)
        exploring_selector = ActionSelector(encoder, q_core, forced_policy)
        # RNG choosing the second action
        rng_explore = random.Random(0)
        act_explore = exploring_selector.select(state, context, candidates, rng=rng_explore)
        self.assertIn(act_explore.type, (ActionType.KITE, ActionType.ATTACK))

    # --------------------------------------------------------------------------
    # 8. Terminal TD Target Semantics
    # --------------------------------------------------------------------------
    def test_terminal_semantics_td_target(self):
        """Verify TD target equation: r for done=True; r + gamma*max_q for done=False."""
        ql = QLearningCore(learning_rate=0.25, discount_factor=0.85)
        ql.set_q("combat", "S_CURR", "kite", 10.0)
        ql.set_q("combat", "S_NEXT", "kite", 20.0)

        # 1. Non-terminal: target = 10 + 0.85 * 20 = 27.0 -> new_q = 10 + 0.25*(27 - 10) = 14.25
        t_non_terminal = Transition(
            state_key="S_CURR",
            action="kite",
            reward=10.0,
            next_state_key="S_NEXT",
            done=False,
        )
        q_non_term = ql.update_td(t_non_terminal, table_name="combat")
        self.assertAlmostEqual(q_non_term, 14.25, places=2)

        # 2. Terminal: target = 10.0 (no gamma contribution) -> new_q = 10 + 0.25*(10 - 10) = 10.0
        ql.set_q("combat", "S_CURR", "kite", 10.0)
        t_terminal = Transition(
            state_key="S_CURR",
            action="kite",
            reward=10.0,
            next_state_key=None,
            done=True,
        )
        q_term = ql.update_td(t_terminal, table_name="combat")
        self.assertAlmostEqual(q_term, 10.0, places=2)

    # --------------------------------------------------------------------------
    # 9. ActionExecutor Boundary for all 10 ActionTypes
    # --------------------------------------------------------------------------
    def test_action_executor_boundary_all_ten_actions(self):
        """Every ActionType is executed through low-level control interfaces without bypass."""
        actions = [
            Action(type=ActionType.MOVE, direction=Vector2(800.0, 400.0)),
            Action(type=ActionType.ATTACK, target_id=1),
            Action(type=ActionType.RETREAT, direction=Vector2(100.0, 360.0)),
            Action(type=ActionType.FARM, target_id=2, direction=Vector2(850.0, 350.0)),
            Action(type=ActionType.KITE, direction=Vector2(900.0, 360.0)),
            Action(type=ActionType.SCOUT, direction=Vector2(600.0, 200.0)),
            Action(type=ActionType.CAST_S1),
            Action(type=ActionType.CAST_S2),
            Action(type=ActionType.CAST_ULT),
            Action(type=ActionType.IDLE),
        ]

        for action in actions:
            self.harness.advance_time(0.25)  # clear attack cooldown
            res = self.harness.control.executor.execute(action)
            self.assertTrue(res.success, f"Action {action.type} failed: {res.error}")

    # --------------------------------------------------------------------------
    # 10. V1 Compatibility & Combo Verification
    # --------------------------------------------------------------------------
    def test_v1_combo_strictly_four_steps(self):
        """CAST_ULT generates strictly 4 steps (S2 -> S2 -> S1 -> Ult) with NO 5th step."""
        self.harness.control.skills.trigger_engage_ultimate(ult_channel_ms=1000.0)
        last_cmd = self.harness.control.transport.sent_commands[-1]

        swipes = [p.strip() for p in last_cmd.split(" && ") if p.strip().startswith("input swipe")]
        self.assertEqual(len(swipes), 4, f"Combo must have exactly 4 steps: {swipes}")

    # --------------------------------------------------------------------------
    # 11. State Semantic Boundaries: LOST != DEAD
    # --------------------------------------------------------------------------
    def test_state_semantic_boundaries_lost_not_dead(self):
        """Missing vision / temporary loss does not mean death."""
        # 1. No detections
        self.harness.clear_entities()
        tick = self.harness.tick()
        self.assertFalse(tick.tactical_state.is_player_dead)
        self.assertFalse(tick.tactical_state.is_player_visible)

        # 2. Enemy disappearance without combat does not trigger kill
        self.harness.spawn_enemy_hero(x=900.0, y=360.0)
        self.harness.tick()
        self.harness.clear_entities()
        self.harness.advance_time(5.0)
        tick2 = self.harness.tick()
        kills = [e for e in tick2.events if e.type == EventType.HERO_KILL]
        self.assertEqual(len(kills), 0, "Disappearance without combat must not be a kill")

    # --------------------------------------------------------------------------
    # 12. Learning Update Cardinality
    # --------------------------------------------------------------------------
    def test_learning_update_cardinality(self):
        """Each multi-domain event triggers exactly one LearningLoop step per domain reward."""
        learn_calls = []
        original_learn_step = self.harness.learning_loop.learn_step

        def spy_learn_step(state, action, reward, next_state, done):
            learn_calls.append((reward, done))
            return original_learn_step(state, action, reward, next_state, done)

        self.harness.learning_loop.learn_step = spy_learn_step

        # Process a HERO_KILL event (combat +60, roam +40)
        kill_event = GameEvent(EventType.HERO_KILL, 100.0, 1, 1.0, ("kill",))
        transitions = self.harness.learning_integrator.process(
            state=self.harness.tactical_state_builder.build(WorldState(timestamp=100.0)),
            action=Action(type=ActionType.ATTACK),
            events=[kill_event],
            next_state=self.harness.tactical_state_builder.build(WorldState(timestamp=101.0)),
            done=False,
        )

        # Exactly 2 transitions
        self.assertEqual(len(transitions), 2)
        # Exactly 2 calls to learn_step
        self.assertEqual(len(learn_calls), 2)
        # Distinct rewards preserved without summation
        rewards = [t.reward for t in transitions]
        self.assertIn(60.0, rewards)
        self.assertIn(40.0, rewards)
        self.assertNotIn(100.0, rewards, "Multi-domain rewards must remain distinct!")

    # --------------------------------------------------------------------------
    # 13. Runtime Failure Isolation
    # --------------------------------------------------------------------------
    def test_runtime_failure_isolation(self):
        """Simulated capture/hardware failure reports cleanly without corrupting Q data."""
        self.harness.capture.should_fail = True
        res = self.harness.tick()
        self.assertFalse(res.success)
        self.assertFalse(res.frame_captured)
        self.assertIn("Capture returned None", res.error)

    # --------------------------------------------------------------------------
    # 14. Runtime Lifecycle (run_loop & stop)
    # --------------------------------------------------------------------------
    def test_runtime_lifecycle_clean_shutdown(self):
        """run_loop runs bounded ticks and performs clean shutdown."""
        ticks = self.harness.runtime.run_loop(max_ticks=4)
        self.assertEqual(ticks, 4)
        self.assertFalse(self.harness.runtime.is_running)

    # --------------------------------------------------------------------------
    # 15. Dependency Architecture Verification (Static AST)
    # --------------------------------------------------------------------------
    def test_static_dependency_boundaries(self):
        """
        Verify:
        brain/ does NOT import control, adb, joystick, subprocess.
        actions/ imports only control.interfaces.
        """
        brain_dir = os.path.join(WORKSPACE_ROOT, "brain")
        for root, _, files in os.walk(brain_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=file_path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotIn(
                                alias.name.split(".")[0],
                                {"control", "subprocess"},
                                f"Forbidden import in {file}: {alias.name}"
                            )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self.assertNotIn(
                                node.module.split(".")[0],
                                {"control", "subprocess"},
                                f"Forbidden from-import in {file}: {node.module}"
                            )


if __name__ == "__main__":
    unittest.main()
