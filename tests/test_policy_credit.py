"""Policy identity must survive control mapping and learning updates."""
import unittest
from actions.models import Action, ActionType
from brain.brain import ClaudeBrainV2
from brain.encoder import TacticalStateEncoder
from brain.learning_loop import LearningLoop
from brain.q_learning import QLearningCore
from brain.transition_builder import TransitionBuilder
from tests.test_brain_v2 import _create_tactical_state


class ExploitRNG:
    def random(self):
        return 1.0

    def choice(self, candidates):
        return candidates[0]


class TestPolicyCredit(unittest.TestCase):
    def setUp(self):
        self.encoder = TacticalStateEncoder()
        self.builder = TransitionBuilder(self.encoder)
        self.q = QLearningCore()
        self.brain = ClaudeBrainV2(q_learning=self.q, random_generator=ExploitRNG())
        self.loop = LearningLoop(self.encoder, self.builder, self.q)

    def test_penalty_changes_next_combat_choice(self):
        state = _create_tactical_state(enemy_visible=True, enemy_count=1, enemy_dist=300.0)
        key = self.encoder.encode(state)
        self.q.set_q('combat', key, 'SWEET_SPOT_BURST', 5)
        self.q.set_q('combat', key, 'KITE_AND_POKE', 1)
        action = self.brain.decide(state)
        self.assertEqual(action.type, ActionType.ATTACK)
        transition = self.loop.learn_step(state, action, -40, None, True)
        self.assertEqual(transition.action, 'SWEET_SPOT_BURST')
        self.assertLess(self.q.get_q('combat', key, 'SWEET_SPOT_BURST'), 1)
        self.assertNotIn('attack', self.q.get_state_actions('combat', key))
        self.assertEqual(self.brain.decide(state).type, ActionType.KITE)

    def test_all_policy_decisions_keep_identity(self):
        state = _create_tactical_state()
        for decision, kind in self.brain.DECISION_TO_ACTION_TYPE.items():
            with self.subTest(decision=decision):
                action = self.brain.map_decision_to_action(decision, state)
                transition = self.builder.build(state, action, 1, None, True)
                self.assertEqual(action.type, kind)
                self.assertEqual(transition.action, decision)

    def test_same_control_command_does_not_collapse_distinct_tactics(self):
        state = _create_tactical_state()
        a = self.brain.map_decision_to_action('LANE_ADVANCE', state)
        b = self.brain.map_decision_to_action('LANE_PUSH', state)
        self.assertEqual(a.type, b.type)
        self.loop.learn_step(state, a, 20, None, True)
        self.loop.learn_step(state, b, -20, None, True)
        values = self.q.get_state_actions('roam', self.encoder.encode(state))
        self.assertEqual(values, {'LANE_ADVANCE': 5.0, 'LANE_PUSH': -5.0})

    def test_plain_actions_and_strings_remain_supported(self):
        state = _create_tactical_state()
        for action, expected in [(Action(ActionType.MOVE), 'move'), ('LANE_PUSH', 'LANE_PUSH')]:
            self.assertEqual(self.builder.build(state, action, 1, None, True).action, expected)


if __name__ == '__main__':
    unittest.main()
