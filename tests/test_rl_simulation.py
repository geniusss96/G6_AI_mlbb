"""Comprehensive test suite for offline synthetic RL simulation.

Verifies:
- Future reward propagation via Temporal-Difference updates
- Negative reward propagation
- Immediate vs future reward trade-offs under different gamma values
- Terminal transition semantics
- Gamma sensitivity (gamma=0.0 vs 0.85 vs 1.0)
- Alpha sensitivity (alpha=0.1 vs 0.5 vs 1.0)
- Convergence of repeated updates
- Greedy candidate action selection in target evaluation
- Complete isolation from ExplorationPolicy
- Strict baseline and production q_brain.json immutability
"""

import hashlib
import os
import unittest
from unittest.mock import patch

from brain.exploration import ExplorationContext, ExplorationPolicy
from brain.q_learning import QLearningCore
from brain.transition import Transition
from tests.rl_simulation.runner import SimulationRunner
from tests.rl_simulation.scenarios import (
    SimulationScenario,
    SyntheticEnvironment,
    SyntheticStep,
    create_greedy_choice_scenario,
    create_immediate_vs_future_scenarios,
    create_negative_future_scenario,
    create_positive_future_scenario,
    create_terminal_scenario,
)


class TestRLSimulation(unittest.TestCase):
    """Offline RL simulation tests in synthetic deterministic environment."""

    def _file_hash(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return ""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def setUp(self) -> None:
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.q_brain_path = os.path.join(self.root_dir, "q_brain.json")
        self.baseline_path = os.path.join(self.root_dir, "data", "q_brain_baseline.json")

        self.initial_q_brain_hash = self._file_hash(self.q_brain_path)
        self.initial_baseline_hash = self._file_hash(self.baseline_path)

    def tearDown(self) -> None:
        # Guarantee disk files remained pristine
        self.assertEqual(self._file_hash(self.q_brain_path), self.initial_q_brain_hash)
        self.assertEqual(self._file_hash(self.baseline_path), self.initial_baseline_hash)

    def test_future_reward_propagation(self) -> None:
        """Scenario 1: Future reward propagates backwards from S1 to S0."""
        runner = SimulationRunner(learning_rate=0.10, discount_factor=0.85)

        # 1. Terminal step in S1 with reward +10.0: S1 -> terminal
        t_s1 = Transition(state_key="S1", action="IDLE", reward=10.0, next_state_key=None, done=True)
        # Repeat S1 so Q(S1, IDLE) converges to 10.0
        runner.run_repeated_transition(t_s1, iterations=70)
        self.assertAlmostEqual(runner.get_q("S1", "IDLE"), 10.0, places=1)

        # 2. Update step in S0 with reward 0.0: S0 --KITE--> S1
        t_s0 = Transition(state_key="S0", action="KITE", reward=0.0, next_state_key="S1", done=False)
        val = runner.step_transition(t_s0)

        # Expected target: 0.0 + 0.85 * 10.0 = 8.5
        # Update: 0.0 + 0.10 * (8.5 - 0.0) = 0.85
        self.assertAlmostEqual(val, 0.85, places=2)
        self.assertAlmostEqual(runner.get_q("S0", "KITE"), 0.85, places=2)
        self.assertGreater(runner.get_q("S0", "KITE"), 0.0)

    def test_negative_reward_propagation(self) -> None:
        """Scenario 2: Future penalty propagates backwards and reduces Q-value."""
        runner = SimulationRunner(learning_rate=0.10, discount_factor=0.85)

        # Prepare S1 with a severe negative value
        runner.set_q("S1", "BAD_ACTION", -20.0)

        # Update S0 --KITE--> S1 with 0 immediate reward
        t_s0 = Transition(state_key="S0", action="KITE", reward=0.0, next_state_key="S1", done=False)
        val = runner.step_transition(t_s0)

        # Target: 0.0 + 0.85 * (-20.0) = -17.0
        # Update: 0.0 + 0.10 * (-17.0 - 0.0) = -1.70
        self.assertAlmostEqual(val, -1.70, places=2)
        self.assertAlmostEqual(runner.get_q("S0", "KITE"), -1.70, places=2)
        self.assertLess(runner.get_q("S0", "KITE"), 0.0)

    def test_immediate_vs_future_reward_tradeoff(self) -> None:
        """Scenario 3: Compare immediate reward (+5) vs delayed reward (+10)."""
        sc_a, sc_b = create_immediate_vs_future_scenarios()

        # Case A: gamma = 0.85 (future is valued: 0.85 * 10 = 8.5 > 5.0)
        runner_high_gamma = SimulationRunner(learning_rate=1.0, discount_factor=0.85)
        # Prepare future state S_B with Q=10.0
        runner_high_gamma.set_q("S_B", "IDLE", 10.0)

        # Step ACTION_A (immediate +5, done)
        runner_high_gamma.run_step(sc_a.steps[0])
        # Step ACTION_B (immediate 0, next S_B)
        runner_high_gamma.run_step(sc_b.steps[0])

        q_a_high = runner_high_gamma.get_q("S0", "ACTION_A")
        q_b_high = runner_high_gamma.get_q("S0", "ACTION_B")

        self.assertAlmostEqual(q_a_high, 5.0, places=2)
        self.assertAlmostEqual(q_b_high, 8.5, places=2)
        self.assertGreater(q_b_high, q_a_high, "With gamma=0.85, delayed reward must exceed immediate reward")

        # Case B: gamma = 0.0 (future is ignored: 0.0 * 10 = 0.0 < 5.0)
        runner_zero_gamma = SimulationRunner(learning_rate=1.0, discount_factor=0.0)
        runner_zero_gamma.set_q("S_B", "IDLE", 10.0)

        runner_zero_gamma.run_step(sc_a.steps[0])
        runner_zero_gamma.run_step(sc_b.steps[0])

        q_a_zero = runner_zero_gamma.get_q("S0", "ACTION_A")
        q_b_zero = runner_zero_gamma.get_q("S0", "ACTION_B")

        self.assertAlmostEqual(q_a_zero, 5.0, places=2)
        self.assertAlmostEqual(q_b_zero, 0.0, places=2)
        self.assertGreater(q_a_zero, q_b_zero, "With gamma=0.0, immediate reward must exceed delayed reward")

    def test_terminal_transition_semantics(self) -> None:
        """Terminal transition (done=True, next_state=None) ignores future Q."""
        runner = SimulationRunner(learning_rate=0.5, discount_factor=0.99)
        sc = create_terminal_scenario()

        val = runner.run_step(sc.steps[0])
        # Initial Q = 0.0, target = 15.0 (reward only, no gamma contribution)
        # new_q = 0.0 + 0.5 * (15.0 - 0.0) = 7.50
        self.assertAlmostEqual(val, 7.50, places=2)
        self.assertAlmostEqual(runner.get_q("S0", "FINISH"), 7.50, places=2)

    def test_gamma_sensitivity(self) -> None:
        """Running identical transition with gamma=0.0, 0.85, 1.0 gives predictable ordering."""
        gammas = [0.0, 0.85, 1.0]
        results = []

        for g in gammas:
            runner = SimulationRunner(learning_rate=1.0, discount_factor=g)
            runner.set_q("S1", "ACTION", 20.0)
            t = Transition(state_key="S0", action="MOVE", reward=0.0, next_state_key="S1", done=False)
            val = runner.step_transition(t)
            results.append(val)

        # Expected:
        # gamma=0.0  -> target = 0.0  -> Q = 0.0
        # gamma=0.85 -> target = 17.0 -> Q = 17.0
        # gamma=1.0  -> target = 20.0 -> Q = 20.0
        self.assertAlmostEqual(results[0], 0.0, places=2)
        self.assertAlmostEqual(results[1], 17.0, places=2)
        self.assertAlmostEqual(results[2], 20.0, places=2)
        self.assertTrue(results[0] < results[1] < results[2])

    def test_alpha_sensitivity(self) -> None:
        """Different learning rates (0.1, 0.5, 1.0) scale step size toward target."""
        alphas = [0.1, 0.5, 1.0]
        updates = []

        for a in alphas:
            runner = SimulationRunner(learning_rate=a, discount_factor=0.85)
            # Terminal target = 20.0, initial Q = 0.0
            t = Transition(state_key="S0", action="ATTACK", reward=20.0, next_state_key=None, done=True)
            val = runner.step_transition(t)
            updates.append(val)

        # alpha=0.1 -> 0 + 0.1 * 20 = 2.0
        # alpha=0.5 -> 0 + 0.5 * 20 = 10.0
        # alpha=1.0 -> 0 + 1.0 * 20 = 20.0
        self.assertAlmostEqual(updates[0], 2.0, places=2)
        self.assertAlmostEqual(updates[1], 10.0, places=2)
        self.assertAlmostEqual(updates[2], 20.0, places=2)

    def test_repeated_learning_convergence(self) -> None:
        """Repeatedly updating the same transition converges asymptotically to TD target."""
        runner = SimulationRunner(learning_rate=0.10, discount_factor=0.85)
        runner.set_q("S1", "BEST", 20.0)

        # Target = 3.0 + 0.85 * 20.0 = 20.0
        t = Transition(state_key="S0", action="STEP", reward=3.0, next_state_key="S1", done=False)
        history = runner.run_repeated_transition(t, iterations=80)

        self.assertEqual(len(history), 80)
        # Convergence: under 2-decimal rounding with alpha=0.1, the update step rounds to 0 when |target - Q| <= 0.05
        self.assertAlmostEqual(history[-1], 20.0, delta=0.1)
        # Monotonically increasing
        for i in range(len(history) - 1):
            self.assertLessEqual(history[i], history[i + 1])

    def test_greedy_target_selection(self) -> None:
        """max Q(s', a') selects the maximum Q-value greedily."""
        sc, prep_steps = create_greedy_choice_scenario()
        runner = SimulationRunner(learning_rate=1.0, discount_factor=0.85)

        # Prepare S1: ACTION_LOW = 5.0, ACTION_HIGH = 25.0
        runner.set_q("S1", prep_steps[0].action, prep_steps[0].reward)
        runner.set_q("S1", prep_steps[1].action, prep_steps[1].reward)

        # Execute S0 --MOVE--> S1 (reward = 2.0)
        val = runner.run_step(sc.steps[0])

        # Target must use 25.0 (greedy max): 2.0 + 0.85 * 25.0 = 2.0 + 21.25 = 23.25
        self.assertAlmostEqual(val, 23.25, places=2)

    def test_exploration_isolation_during_td_update(self) -> None:
        """update_td does not invoke ExplorationPolicy.should_explore."""
        runner = SimulationRunner(learning_rate=0.10, discount_factor=0.85)
        runner.set_q("S1", "ACT", 10.0)
        t = Transition(state_key="S0", action="ACT", reward=5.0, next_state_key="S1", done=False)

        with patch.object(ExplorationPolicy, "should_explore", autospec=True) as mock_explore:
            runner.step_transition(t)
            self.assertEqual(mock_explore.call_count, 0, "ExplorationPolicy must never be called by update_td")

    def test_synthetic_environment_stepping(self) -> None:
        """Tests SyntheticEnvironment determinism and step sequence."""
        env = SyntheticEnvironment(initial_state="S0")
        env.add_transition(state="S0", action="FORWARD", reward=1.0, next_state="S1", done=False)
        env.add_transition(state="S1", action="GOAL", reward=10.0, next_state=None, done=True)

        step1 = env.step("FORWARD")
        self.assertEqual(step1.state, "S0")
        self.assertEqual(step1.action, "FORWARD")
        self.assertEqual(step1.reward, 1.0)
        self.assertEqual(step1.next_state, "S1")
        self.assertFalse(step1.done)

        step2 = env.step("GOAL")
        self.assertEqual(step2.state, "S1")
        self.assertEqual(step2.action, "GOAL")
        self.assertEqual(step2.reward, 10.0)
        self.assertIsNone(step2.next_state)
        self.assertTrue(step2.done)

        # Once done, step raises RuntimeError until reset
        with self.assertRaises(RuntimeError):
            env.step("GOAL")

        env.reset()
        self.assertEqual(env.current_state, "S0")


if __name__ == "__main__":
    unittest.main()
