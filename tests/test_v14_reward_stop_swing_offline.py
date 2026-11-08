"""Offline checks: no AnyLogic, PyTorch or checkpoint required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from thesis.envs.v14_reward_shaping import RewardConfig, RoutingRewardShaper


def observation(x=0.5, y=0.5, mask=(1, 1, 0, 0, 0), near=None):
    agvs = np.zeros((20, 38), dtype=float)
    agvs[0, 0] = 1
    agvs[0, 2:4] = [x, y]
    agvs[0, 8:10] = [x + 0.1, y]
    if near is not None:
        agvs[1, 0] = 1
        agvs[1, 2:4] = near
        agvs[1, 4:6] = near
    return dict(agvs=agvs, stat=np.zeros((27, 38)),
                action_mask=np.asarray(mask, dtype=float))


class V14DirectRewardTests(unittest.TestCase):
    def setUp(self):
        self.s = RoutingRewardShaper()

    def test_stop_with_clear_legal_move_penalized_only_after_next_observation(self):
        o = observation()
        self.assertEqual(self.s.observe(3, o, 2.0), 2.0)
        self.s.choose(3, 0)
        self.assertEqual(self.s.observe(3, o, 2.0), 1.5)
        self.assertEqual(self.s.report()['stop_penalties_applied'], 1)
        self.assertEqual(self.s.observe(3, o, 2.0), 2.0)

    def test_wait_with_no_move_is_not_penalized(self):
        o = observation(mask=(1, 0, 0, 0, 0))
        self.s.observe(1, o, 2.0)
        self.s.choose(1, 0)
        self.assertEqual(self.s.observe(1, o, 2.0), 2.0)
        self.assertEqual(self.s.report()['protected_wait_no_legal_move'], 1)

    def test_wait_near_another_agv_is_protected(self):
        o = observation(near=(0.51, 0.5))
        self.s.observe(0, o, 1.0)
        self.s.choose(0, 0)
        self.assertEqual(self.s.observe(0, o, 1.0), 1.0)
        self.assertEqual(self.s.report()['protected_wait_nearby_agv'], 1)

    def test_wait_if_candidate_node_contested_is_protected(self):
        o = observation(near=(0.6, 0.5))
        self.s.observe(0, o, 0.0)
        self.s.choose(0, 0)
        self.assertEqual(self.s.observe(0, o, 0.0), 0.0)
        self.assertEqual(self.s.report()['protected_wait_no_clear_candidate'], 1)

    def test_return_to_previous_node_is_swing(self):
        a = observation(x=0.3)
        b = observation(x=0.4)
        self.assertEqual(self.s.observe(4, a, 0.0), 0.0)
        self.s.choose(4, 1)
        self.assertEqual(self.s.observe(4, b, 0.0), 0.0)
        self.s.choose(4, 2)
        self.assertEqual(self.s.observe(4, a, 0.0), -1.0)
        self.assertEqual(self.s.report()['swing_penalties_applied'], 1)

    def test_no_swing_for_stop_at_same_coordinate(self):
        a = observation(x=0.3, mask=(1, 0, 0, 0, 0))
        b = observation(x=0.4, mask=(1, 0, 0, 0, 0))
        self.s.observe(4, a, 0.0)
        self.s.choose(4, 1)
        self.s.observe(4, b, 0.0)
        self.s.choose(4, 0)
        self.assertEqual(self.s.observe(4, b, 0.0), 0.0)
        self.assertEqual(self.s.report().get('swing_penalties_applied', 0), 0)

    def test_episode_reset_erases_physical_agv_state(self):
        self.s.observe(1, observation(), 0.0)
        self.s.choose(1, 0)
        self.s.reset_episode()
        self.assertEqual(self.s.observe(1, observation(), 0.0), 0.0)
        self.assertEqual(self.s.report().get('stop_penalties_applied', 0), 0)

    def test_reward_is_routing_only_in_installed_behavior(self):
        class DummyMultiAgent:
            def __init__(self, dispatching):
                self.dispatching = dispatching
            def convert_from_observation(self, alpyne_obs):
                return observation(), float(alpyne_obs.rew), False, {}
            def convert_to_action(self, action, agent):
                return ('original', action, agent)
        module = SimpleNamespace(MultiAgent=DummyMultiAgent)
        self.s.install(module)
        routing = DummyMultiAgent(False)
        dispatch = DummyMultiAgent(True)
        self.assertEqual(dispatch.convert_from_observation(SimpleNamespace(caller=1001, rew=4))[1], 4)
        self.assertEqual(dispatch.convert_to_action(0, '1001'), ('original', 0, '1001'))
        self.assertEqual(routing.convert_from_observation(SimpleNamespace(caller=1, rew=4))[1], 4)
        self.assertEqual(routing.convert_to_action(0, '1'), ('original', 0, '1'))
        self.assertEqual(routing.convert_from_observation(SimpleNamespace(caller=1, rew=4))[1], 3.5)
        self.assertEqual(dispatch.convert_from_observation(SimpleNamespace(caller=1001, rew=4))[1], 4)

    def test_positive_penalty_forbidden(self):
        with self.assertRaises(ValueError):
            RewardConfig(stop_penalty=0.1)


if __name__ == '__main__':
    unittest.main()
