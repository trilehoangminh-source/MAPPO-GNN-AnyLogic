"""Offline sanity tests for Tobias B3 graph prior, no Java or Alpyne required."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from types import SimpleNamespace
import numpy as np
import torch

from thesis.mappo.routing_guidance_v121 import DirectedGraphGuide, GuidedMAPPO
from thesis.mappo.algorithm import MAPPOConfig


def sample_observation():
    ego = np.zeros((8, 51), dtype=np.float32)
    ego[0, 4:6] = [0, 0]    # current
    ego[0, 6:8] = [2, 0]    # target
    ego[0, 8:16] = [1, 0, 0, 1, 0, 0, 0, 0]  # two legal directions
    return {'agvs': ego, 'stat': np.zeros((27,51), dtype=np.float32),
            'action_mask': np.array([0,1,1,0,0], dtype=np.float32)}


def make_graph():
    nodes = OrderedDict([(0,(0.,0.)), (1,(1.,0.)), (2,(0.,1.)),
                         (3,(2.,0.)), (4,(2.,1.))])
    return SimpleNamespace(nodes=nodes, paths=[(0,1), (1,3), (0,2), (2,4),(4,3)])


def test_graph_and_directed_paths():
    hive = make_graph()
    guide = DirectedGraphGuide(hive, strength=2.0)
    obs = sample_observation()
    prior, status = guide.suggest(obs)
    assert status == 'guided', (prior, status)
    assert np.array_equal(prior, np.array([0,2,0,0,0], np.float32)), prior
    # No reverse arc allowed: map 0->1 replaced by 1->0, so choose action2.
    hive.paths = [(1,0),(1,3),(0,2),(2,4),(4,3)]
    prior, status = guide.suggest(obs)
    assert status == 'guided' and prior[2] == 2 and prior[1] == 0
    # Missing graph must not silently invent a route.
    hive.paths = None
    prior, status = guide.suggest(obs)
    assert status == 'missing_graph' and np.count_nonzero(prior) == 0


def test_ppo_distribution_and_gradients():
    feature = 51
    obs_space = {'agvs':SimpleNamespace(shape=(8,feature)),
                 'stat':SimpleNamespace(shape=(27,feature)),
                 'action_mask':SimpleNamespace(shape=(5,))}
    cfg=MAPPOConfig(routing_backbone='pooled', dispatch_backbone='pooled',hidden=8)
    model=GuidedMAPPO({'routing':(obs_space,5)},feature,cfg)
    obs=sample_observation()
    obs['routing_guide_bias']=np.array([0,3,0,0,0],np.float32)
    actor=model.roles['routing'].actor
    actor.eval()
    prepared=__import__('thesis.mappo.algorithm',fromlist=['as_actor']).as_actor(obs,'cpu')
    input_dict={k:v.unsqueeze(0) for k,v in prepared.items()}
    with torch.no_grad():
        p=actor.dist(input_dict).probs[0]
        assert p[1] > p[2], p
        assert p[0]==0 and p[3]==0 and p[4]==0, p
        orig=actor.dist(input_dict).log_prob(torch.tensor([1]))[0]
    action, logp, val=model.act('routing',obs,{'agvs':obs['agvs'],'stat':obs['stat'],'ego':0},greedy=True)
    assert action==1 and abs(logp-orig.item()) < 1e-5
    # Masked action remains zero even with a huge illegal prior.
    obs['routing_guide_bias'][4]=20
    inp={k:v.unsqueeze(0) for k,v in __import__('thesis.mappo.algorithm',fromlist=['as_actor']).as_actor(obs,'cpu').items()}
    assert actor.dist(inp).probs[0,4] == 0
    # Update path must preserve bias (same stored prior), with valid logp.
    actor.train()
    act, old_lp, val=model.act('routing',obs,{'agvs':obs['agvs'],'stat':obs['stat'],'ego':0},greedy=False)
    trajectories=[[dict(role='routing',observation=obs,state={'agvs':obs['agvs'],
                      'stat':obs['stat'],'ego':0},action=act,logp=old_lp,value=val,
                      next_value=0.,reward=1.,done=True)]]
    report=model.train(trajectories)
    assert report['routing']['transitions']==1 and np.isfinite(report['routing']['actor_loss'])


if __name__=='__main__':
    test_graph_and_directed_paths()
    test_ppo_distribution_and_gradients()
    print('PASS: directed graph routing, soft legal preference, PPO rollout/update distribution parity')
