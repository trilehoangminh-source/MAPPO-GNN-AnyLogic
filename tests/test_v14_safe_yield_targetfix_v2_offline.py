"""V2 regression tests for corrected target indices; no AnyLogic/GPU required."""
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thesis.envs.v14_safe_yield_stall import SafeYieldStallReward
from scripts.train_v14_safe_yield_stall_targetfix_v2 import assert_target_fix


def fleet(last=(.5,.5), nxt=(.6,.5), target=(.8,.5)):
    a=np.zeros((20,38),dtype=float)
    a[0,0]=1.0
    a[0,2:4]=last
    a[0,4:6]=nxt
    a[0,6:8]=target
    return a


class TargetIndexFixTests(unittest.TestCase):
    def test_guard_corrected_source(self):
        self.assertTrue(assert_target_fix())

    def test_next_node_change_must_not_reset_target(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(nxt=(.6,.5),target=(.8,.5)),0,0)
        r.choose(0,0)
        value=r.observe(0,fleet(nxt=(.7,.5),target=(.8,.5)),60,0)
        self.assertAlmostEqual(value, -.15)
        self.assertEqual(r.report().get('target_changes_reset_dwell',0),0)
        self.assertEqual(r.report().get('stall_penalties_applied',0),1)

    def test_real_target_change_resets_dwell(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(target=(.8,.5)),0,0)
        r.choose(0,0)
        value=r.observe(0,fleet(target=(.9,.5)),100,0)
        self.assertEqual(value,0)
        self.assertEqual(r.report().get('target_changes_reset_dwell',0),1)

    def test_wait_at_target_exempt(self):
        r=SafeYieldStallReward()
        r.observe(0,fleet(last=(.8,.5),target=(.8,.5)),0,0)
        r.choose(0,0)
        self.assertEqual(r.observe(0,fleet(last=(.8,.5),target=(.8,.5)),100,0),0)
        self.assertEqual(r.report().get('stall_penalties_applied',0),0)


if __name__=='__main__': unittest.main()
