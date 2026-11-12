"""Smoke test for V15 routing without action mask (no AnyLogic needed).

Run from project root: python test_v15_routing_no_mask.py
Place this file at project root; it imports the two V15 files in thesis/mappo.
"""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from thesis.mappo.v15_tobias_arch import TobiasGraphRoutingActor, TobiasLinearDispatchActor
from thesis.mappo.collector_stop_ab import EventCollector

class Graph:
    xy=((.1,.1),(.2,.1),(.2,.2))
    arcs=((0,1),(1,2),(2,0))
    fingerprint='test'

class Space:
    def __init__(self,shape): self.shape=shape

def observation():
    agvs=torch.zeros((2,20,19))
    stations=torch.zeros((2,27,19))
    agvs[:,0,2:16]=torch.tensor([.1,.1,.1,.1,.2,.2,.2,.1,.1,.1,.2,.2,.1,.1])
    agvs[:,0,1]=1
    agvs[:,0,18]=.8
    stations[:,0,:2]=torch.tensor([.2,.1])
    return {'agvs':agvs,'stat':stations,'action_mask':torch.tensor([[1.,1,0,0,0],[1.,1,0,0,0]])}

class FakeEnvironment:
    def __init__(self,role):
        self.role=role
        self.routing_agent_death=False
        self.dispatching_agent_death=False
        self.agent_selection='0' if role=='routing' else '1000'
        self.agents=['0']
        self.statistics={}
        self.at_end=False
        self.last_action=None
    def reset(self,seed=None):
        self.at_end=False
        self.agents=['0']
    def last(self):
        obs={'agvs':np.zeros((20,19),np.float32),
             'stat':np.zeros((27,19),np.float32),
             'action_mask': np.asarray([1,1,0,0,0] if self.role=='routing' else [1,1,0,0,0]+[0]*22,np.float32)}
        return obs,(1. if self.at_end else 0.),self.at_end,{}
    def get_mappo_state(self):return {'ego':0,'agvs':np.zeros((20,19)),'stat':np.zeros((27,19))}
    def get_mappo_time(self):return 1. if self.at_end else 0.
    def step(self,action):
        self.last_action=action
        self.at_end=True

class FakeModel:
    def __init__(self,role,action):
        self.role=role;self.action=action
        self.roles={role:SimpleNamespace(actor=SimpleNamespace(action_dim=5 if role=='routing' else 27))}
        self.cfg=SimpleNamespace(two_fleets=True)
    def role_for(self,*args):return self.role
    def value(self,*args):return 0.
    def act(self,*args,**kwargs):return self.action,0.,0.

class NoMaskTest(unittest.TestCase):
    def test_routing_ignores_mask_but_dispatch_respects_it(self):
        torch.manual_seed(2)
        routing=TobiasGraphRoutingActor(Graph()).eval()
        o=observation()
        with torch.no_grad():
            p=routing.dist(o).probs
            expected=routing.logits(o).softmax(-1)
        self.assertTrue(torch.allclose(p,expected))
        self.assertTrue(torch.all(p[:,2:5]>0))
        dispatch=TobiasLinearDispatchActor({'agvs':Space((20,19)),'stat':Space((27,19))},27).eval()
        d=observation()
        d['action_mask']=torch.ones((2,27))
        d['action_mask'][:,13]=0
        with torch.no_grad(): dp=dispatch.dist(d).probs
        self.assertTrue(torch.all(dp[:,13]==0))

    def test_collector_does_not_reject_invalid_routing(self):
        env=FakeEnvironment('routing')
        _segments,meta=EventCollector(FakeModel('routing',3),env).episode(train=True)
        self.assertEqual(env.last_action,3)
        self.assertEqual(meta['action_diagnostics']['routing_invalid_selected'],1)
        self.assertEqual(meta['action_diagnostics']['selected_mask_violations'],1)

    def test_dispatch_still_rejects_invalid_action(self):
        env=FakeEnvironment('dispatch_front')
        with self.assertRaisesRegex(RuntimeError,'violates mask'):
            EventCollector(FakeModel('dispatch_front',3),env).episode(train=True)

if __name__=='__main__':unittest.main(verbosity=2)
