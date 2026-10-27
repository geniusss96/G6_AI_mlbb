import unittest
from vision.detector import Detection
from world.builder import WorldStateBuilder
from brain.state import TacticalStateBuilder


def detection(name, x=800):
    return Detection(class_id=0, class_name=name, confidence=.9,
                     x1=x-10, y1=350, x2=x+10, y2=370)


class TestWorldClassSemantics(unittest.TestCase):
    def test_enemy_substring_does_not_turn_minions_and_turrets_into_heroes(self):
        world = WorldStateBuilder().build(10, detections=[
            detection('hp_enemy'), detection('minion_enemy'), detection('turret_enemy'),
            detection('minion_ally'), detection('turret_ally'), detection('red_baff'),
            detection('jungle-monstr'), detection('hp_self', 772)])
        self.assertEqual(len(world.enemies), 1)
        self.assertEqual(len(world.minions), 4)
        self.assertEqual(len(world.turrets), 2)
        self.assertEqual([m.is_enemy for m in world.minions], [True, False, True, True])
        self.assertEqual([t.is_enemy for t in world.turrets], [True, False])
        tactical = TacticalStateBuilder().build(world)
        self.assertEqual(tactical.enemy_count, 1)
        self.assertEqual(tactical.minion_count, 3)
        self.assertFalse(world.player.hp.visible)
        self.assertEqual(world.player.hp.confidence, 0)

    def test_allied_base_is_not_player(self):
        world = WorldStateBuilder().build(10, detections=[detection('tron_self')])
        self.assertFalse(world.player.is_visible)
        world = WorldStateBuilder().build(10, detections=[detection('hero_self')])
        self.assertTrue(world.player.is_visible)
