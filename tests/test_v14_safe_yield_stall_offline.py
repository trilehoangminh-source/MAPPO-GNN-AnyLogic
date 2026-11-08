"""Offline unit tests; do not require Java, GPU, or old model checkpoint."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thesis.envs.v14_safe_yield_stall import SafeYieldStallReward,SafeYieldConfig


def fleet(ego_x=.5,ego_goal=.8,other_x=None):
    a=np.zeros((20,38),dtype=float)
    a[0,0]=1;a[0,2:4]=[ego_x,.5];a[0,4:6]=[ego_goal,.5]
    if other_x is not None:
        a[1,0]=1;a[1,2:4]=[other_x,.5];a[1,4:6]=[other_x,.5]
    return a

class RewardTests(unittest.TestCase):
    def test_no_blanket_stop_penalty_and_clock_grace(self):
        r=SafeYieldStallReward()
        self.assertEqual(r.observe(0,fleet(),0,3),3)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(),40,3),3)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(),65,3),2.8)

    def test_stall_charged_only_for_incremental_sim_seconds(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(),0,0);r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(),55,0),-.1)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(),55,0),0)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(),65,0),-.1)

    def test_stall_cap_and_movement_reset(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(),0,0);r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(),1000,0),-2)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(),1200,0),0)
        r.choose(0,1);r.observe(0,fleet(ego_x=.6),1210,0)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(ego_x=.6),1270,0),-.15)

    def test_at_target_processing_is_exempt(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(ego_goal=.5),0,0);r.choose(0,0)
        self.assertEqual(r.observe(0,fleet(ego_goal=.5),1000,0),0)
        self.assertEqual(r.report().get('stall_penalties_applied',0),0)

    def test_yield_bonus_only_after_other_moved_away(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(other_x=.53),10,1);r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(other_x=.58),20,1),1.35)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(other_x=.8),25,1),1)
        self.assertEqual(r.report()['yield_proxy_bonuses_applied'],1)

    def test_no_bonus_for_other_still_near_or_ego_moved(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(other_x=.53),10,0);r.choose(0,0)
        self.assertEqual(r.observe(0,fleet(other_x=.53),12,0),0)
        r.choose(0,0)
        self.assertEqual(r.observe(0,fleet(ego_x=.6,other_x=.63),14,0),0)

    def test_new_dispatch_target_resets_long_processing_dwell(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(ego_goal=.5),0,0);r.choose(0,0)
        r.observe(0,fleet(ego_goal=.5),1600,0);r.choose(0,0)
        # Destination changed while the vehicle is still at the workstation.
        self.assertEqual(r.observe(0,fleet(ego_goal=.8),1601,0),0)
        r.choose(0,0)
        self.assertAlmostEqual(r.observe(0,fleet(ego_goal=.8),1651,0),-.05)
        self.assertEqual(r.report()['target_changes_reset_dwell'],1)

    def test_other_agv_identity_is_physical_not_shuffled(self):
        r=SafeYieldStallReward()
        old=fleet(other_x=.53)
        r.observe(0,old,10,0);r.choose(0,0)
        current=fleet(other_x=.58)
        self.assertEqual(r.observe(0,current,20,0),.35)

    def test_swing_detection(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(.3),0,0);r.choose(0,1)
        r.observe(0,fleet(.4),5,0);r.choose(0,2)
        self.assertEqual(r.observe(0,fleet(.3),10,0),-1)

    def test_no_inter_episode_leak(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(),0,0);r.choose(0,0)
        r.reset_episode()
        self.assertEqual(r.observe(0,fleet(),3600,7),7)
        self.assertEqual(r.report().get('stall_penalties_applied',0),0)

    def test_invalid_coefficients_refused(self):
        with self.assertRaises(ValueError): SafeYieldConfig(stall_penalty_per_10s=1)
        with self.assertRaises(ValueError): SafeYieldConfig(yield_bonus=-1)

    def test_environment_save_hook_routing_only_not_transition_rewrite(self):
        class FakeBehavior:
            dispatching=False
            def convert_to_action(self,actions,agent):
                return ('original_action',actions,agent)
        class FakeMatrix:
            def __init__(self):
                self.rewards={};self.agv=fleet();self.t=0
            def _save_observation(self, alpyne_obs,agent,obs,reward,done,info):
                self.rewards[agent]=reward
            def get_mappo_state(self):
                return {'agvs':self.agv}
            def get_mappo_time(self):
                return self.t
        bmod=SimpleNamespace(MultiAgent=FakeBehavior)
        mmod=SimpleNamespace(Matrix=FakeMatrix)
        r=SafeYieldStallReward()
        r.install(bmod,mmod)
        inst=FakeMatrix();behavior=FakeBehavior()
        obs=SimpleNamespace(caller=0)
        inst._save_observation(obs,'0',None,5,False,{})
        self.assertEqual(inst.rewards['0'],5)
        behavior.convert_to_action(0,'0')
        inst.t=60;inst._save_observation(obs,'0',None,5,False,{})
        self.assertAlmostEqual(inst.rewards['0'],4.85)
        inst._save_observation(SimpleNamespace(caller=1000),'1000',None,8,False,{})
        self.assertEqual(inst.rewards['1000'],8)
        with self.assertRaises(RuntimeError):r.install(bmod,mmod)

if __name__=='__main__':unittest.main()
