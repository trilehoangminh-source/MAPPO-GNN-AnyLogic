"""Offline CPU tests for new B3-GNN; run from project root:
python tests/test_gnn_B3_v121.py
No AnyLogic or model ZIP needed; REQUIRES existing MAPPO B3 source files.
"""
from __future__ import annotations
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec, GraphRoutingActor, upgrade_guided_mappo_to_gnn
from thesis.mappo.routing_guidance_v121 import DirectedGraphGuide, GuidedMAPPO
from thesis.mappo.algorithm import MAPPOConfig, as_actor, batchify


def make_hive():
    return SimpleNamespace(nodes=OrderedDict([(0,(0.,0.)),(1,(1.,0.)),(2,(0.,1.)),
                                              (3,(2.,0.)),(4,(2.,1.))]),
                           paths=[(0,1),(1,3),(0,2),(2,4),(4,3)])


def sample_obs():
    agvs=np.zeros((8,51), dtype=np.float32)
    agvs[0,0]=1.
    agvs[0,4:6]=[0.,0.]
    agvs[0,6:8]=[2.,0.]
    agvs[0,8:16]=[1,0,0,1,0,0,0,0]
    agvs[1,0]=1.
    agvs[1,4:6]=[0.,1.]
    return {'agvs':agvs, 'stat':np.zeros((27,51),np.float32),
            'action_mask':np.array([0,1,1,0,0],np.float32)}


def space():
    return {'agvs':SimpleNamespace(shape=(8,51)),
            'stat':SimpleNamespace(shape=(27,51)),
            'action_mask':SimpleNamespace(shape=(5,))}


def test_topology_and_gradients():
    torch.manual_seed(14)
    hive=make_hive()
    spec=GraphSpec.from_hive(hive)
    assert spec.n == 5 and len(spec.arcs) == 5
    obs={k:torch.as_tensor(v).unsqueeze(0) for k,v in sample_obs().items()}
    actor=GraphRoutingActor(space(),spec,hidden=8,layers=2)
    d=actor.dist(obs)
    assert d.probs.shape==(1,5)
    assert d.probs[0,0]==0 and d.probs[0,3]==0 and d.probs[0,4]==0
    assert abs(float(d.probs.sum())-1)<1e-6
    loss=-d.log_prob(torch.tensor([1]))
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.layers[0].parameters() if p.grad is not None)>0, 'GNN path has no gradient'
    assert sum(float(p.grad.abs().sum()) for p in actor.node_in.parameters() if p.grad is not None)>0
    # Directed topology alters node representations/logits with identical weights.
    rev=make_hive()
    rev.paths=[(b,a) for a,b in hive.paths]
    reverse=GraphRoutingActor(space(),GraphSpec.from_hive(rev),hidden=8,layers=2)
    reverse.load_state_dict(actor.state_dict()) # also copies adjacency! restore reverse adjacency explicitly
    reverse.outgoing_average.copy_(GraphRoutingActor(space(),GraphSpec.from_hive(rev),hidden=8,layers=2).outgoing_average)
    with torch.no_grad():
        assert not torch.allclose(actor.logits(obs),reverse.logits(obs)), 'Directed edges have no effect'
    print('PASS graph shape, hard mask, directed messages, nonzero GNN gradients')


def test_mappo_ppo_same_distribution_and_optimizer():
    torch.manual_seed(43)
    obs_space=space()
    cfg=MAPPOConfig(routing_backbone='pooled',dispatch_backbone='pooled',hidden=8,epochs=2,mini_batches=1)
    model=GuidedMAPPO({'routing':(obs_space,5)},51,cfg,device='cpu')
    spec=upgrade_guided_mappo_to_gnn(model,make_hive(),hidden=8,layers=2)
    assert spec.n==5
    role=model.roles['routing']
    assert isinstance(role.actor,GraphRoutingActor)
    # Verify the optimizer tracks the NEW GNN parameters (not the discarded pooled actor).
    actor_param_ids={id(p) for p in role.actor.parameters()}
    optimizer_param_ids={id(p) for group in role.actor_optimizer.param_groups for p in group['params']}
    assert actor_param_ids==optimizer_param_ids
    obs=sample_obs()
    bias,status=DirectedGraphGuide(make_hive(),strength=1.5).suggest(obs)
    assert status=='guided' and bias[1]==1.5
    obs['routing_guide_bias']=bias
    state={'agvs':np.zeros((8,51),np.float32), 'stat':np.zeros((27,51),np.float32),'ego':0}
    act,old_lp,value=model.act('routing',obs,state,greedy=False)
    # Check same rollout vs PPO distribution, including mask and Dijkstra prior.
    import thesis.mappo.algorithm as algo
    dist=role.actor.dist(algo.batchify(algo.as_actor(obs,torch.device('cpu'))))
    assert abs(float(dist.log_prob(torch.tensor([act])))-old_lp)<1e-6
    assert obs['action_mask'][act]>0.5
    params_before={k:v.detach().clone() for k,v in role.actor.state_dict().items()}
    meta=model.train([[dict(role='routing',observation=obs,state=state,
                            action=act,logp=old_lp,value=value,reward=2.,
                            next_value=0.,done=True)]])
    assert meta['routing']['optimizer_steps']==2
    diffs=[(p-params_before[k]).abs().sum().item() for k,p in role.actor.state_dict().items()
           if p.is_floating_point()]
    assert max(diffs)>0, 'GNN actor not updated by PPO optimizer'
    print('PASS B3+GNN logprob parity, trainable GNN optimizer, PPO step')

if __name__=='__main__':
    test_topology_and_gradients()
    test_mappo_ppo_same_distribution_and_optimizer()
    print('ALL B3-GNN OFFLINE TESTS PASSED')
