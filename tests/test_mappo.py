"""Run with: python -m unittest discover -s tests -p 'test_mappo.py' -v"""
import sys
import unittest
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thesis.mappo.algorithm import MAPPO, MAPPOConfig
from thesis.mappo.collector import EventCollector
from thesis.mappo.networks import masked_logits


class FakeSpace:
    def __init__(self, shape):
        self.shape = shape


def obs_space():
    return {'agvs':FakeSpace((4, 20)), 'stat':FakeSpace((3, 20))}


def fake_actor_obs(action_count):
    a = np.zeros((4, 20), np.float32)
    a[0, :3] = [1,.2,.3]
    a[1, :3] = [1,.3,.5]
    s = np.zeros((3, 20), np.float32)
    s[0, :3] = [.4,.5,1]
    return {'agvs':a, 'stat':s, 'action_mask':np.ones((action_count,), np.float32)}


def fake_state(ego):
    o=fake_actor_obs(5)
    return {'agvs':o['agvs'], 'stat':o['stat'], 'ego':int(ego)}


class MockAEC:
    """Sequence R0,D0,R0,D0,R0(term), rewards given to SAME role."""
    def __init__(self):
        self.frames=[('0',0,0,False),('1000',0,0,False),('0',0,2,False),
                     ('1000',0,3,False),('0',0,4,True)]
        self.pos=0
        self.agents=['0','1000']
        self.agent_selection='0'
        self.statistics={'products':2}

    def reset(self,seed=None):
        self.pos=0
        self.agents=['0','1000']
        self.agent_selection='0'

    def last(self):
        who,ego,reward,done=self.frames[self.pos]
        if done:
            self.agents=[]
        return fake_actor_obs(7),reward,done,{}

    def get_mappo_state(self):
        return fake_state(self.frames[self.pos][1])

    def get_mappo_time(self):
        return float(self.pos)*5

    def step(self,action):
        assert self.agents
        self.pos+=1
        self.agent_selection=self.frames[self.pos][0]


class TestMAPPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cfg=MAPPOConfig(hidden=16, epochs=2, routing_backbone='pooled',
                        dispatch_backbone='pooled',two_fleets=False)
        cls.model=MAPPO({'routing':(obs_space(),5), 'dispatching':(obs_space(),7)},20,cfg)

    def test_masking_blocks_invalid_actions(self):
        logits=torch.zeros(1,7)
        mask=torch.tensor([[1,0,1,0,0,0,0]],dtype=torch.float32)
        dist=torch.distributions.Categorical(logits=masked_logits(logits,mask))
        self.assertEqual(float(dist.probs[0,1]),0.0)
        self.assertEqual(float(dist.probs[0,6]),0.0)

    def test_rollout_closes_correct_agent(self):
        trajectories,meta=EventCollector(self.model,MockAEC()).episode(seed=1)
        self.assertEqual(meta['closed_transitions'],3)
        self.assertEqual(meta['discarded_pending'],1)
        reward_by_role={(x[0]['role']):[s['reward'] for s in x] for x in trajectories}
        self.assertEqual(reward_by_role['routing'],[2.0,4.0])
        self.assertEqual(reward_by_role['dispatching'],[3.0])
        self.assertTrue(trajectories[0][-1]['done'] or trajectories[1][-1]['done'])
        # End-to-end transitions are returned again in the next test.

    def test_update_is_finite(self):
        trajectories,_=EventCollector(self.model,MockAEC()).episode(seed=1)
        report=self.model.train(trajectories)
        self.assertEqual(set(report),{'routing','dispatching'})
        for role,r in report.items():
            self.assertTrue(np.isfinite(r['actor_loss']))
            self.assertTrue(np.isfinite(r['value_loss']))
            self.assertEqual(r['optimizer_steps'],2)

    def test_checkpoint(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pt'
            self.model.save(path)
            self.assertTrue(path.exists())
            self.model.load(path)

    def test_role_mapping(self):
        self.assertEqual(self.model.role_for('0',0,True),'routing')
        self.assertEqual(self.model.role_for('1007',2,True),'dispatch_front')
        self.assertEqual(self.model.role_for('1007',3,True),'dispatch_rear')

    def test_critic_independent_of_other_agv_order(self):
        model=self.model.roles['routing'].critic
        x=fake_state(0)
        from thesis.mappo.algorithm import as_global,batchify
        with torch.no_grad():
            v1=model(batchify(as_global(x,torch.device('cpu')))).item()
            x['agvs'][1],x['agvs'][2] = x['agvs'][2].copy(),x['agvs'][1].copy()
            v2=model(batchify(as_global(x,torch.device('cpu')))).item()
        self.assertAlmostEqual(v1,v2,places=5)

if __name__=='__main__':unittest.main()
