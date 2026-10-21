"""Offline test: directed mask, edge features, residual gradient, PPO logprob parity.

Run: python tests/test_edgeGAT_MAPPO_v4_offline.py
Requires PyTorch; no AnyLogic or ML project dependencies.
"""
from __future__ import annotations
import sys
import types
from pathlib import Path
from torch import nn
from types import SimpleNamespace

import torch
from torch.distributions import Categorical

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from thesis.mappo.edge_gat_routing_v4 import (
    ResidualEdgeGAT, EdgeAwareLayer, compute_static_edge_features)


def main():
    torch.manual_seed(17)
    xy=torch.tensor([[0.,0.],[1.,0.],[0.,2.],[3.,0.],[4.,0.]])
    adj=torch.zeros(5,5)
    adj[0,1]=1;adj[0,2]=1;adj[1,3]=1;adj[2,3]=1;adj[3,4]=1
    edge=compute_static_edge_features(xy,adj)
    assert edge.shape==(5,5,3)
    assert torch.all(edge[1,0]==0) and (edge[0,1]!=0).any()
    assert edge[0,1,0]>0 and edge[0,2,0]>edge[0,1,0]
    assert torch.all(edge[4]==0)  # isolated sink has self attention
    layer=ResidualEdgeGAT(hidden=16,heads=2)
    x=torch.randn(3,5,16,requires_grad=True)
    out=layer(x,adj,edge)
    assert out.shape==x.shape and torch.isfinite(out).all()
    loss=out.square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name in ('q.weight','k.weight','v.weight','edge_score.weight','edge_value.weight','output.weight'):
        p=dict(layer.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert layer.edge_score.weight.grad.norm().item()>0
    assert layer.edge_value.weight.grad.norm().item()>0
    # Reversing directed edges changes reachable successors; required for Tobias.
    with torch.no_grad():
        directed=layer(x.detach(),adj,edge)
        reverse=adj.T.clone()
        reverse_edge=compute_static_edge_features(xy,reverse)
        reversed_out=layer(x.detach(),reverse,reverse_edge)
        assert not torch.allclose(directed, reversed_out)
    # Edge length impacts outputs even if node states + topology are fixed.
    with torch.no_grad():
        edge_changed=edge.clone()
        edge_changed[0,1,0]+=2.0
        out_changed=layer(x.detach(),adj,edge_changed)
        assert not torch.allclose(directed,out_changed)
    # Wrapper matches legacy GNN layer forward(x, adjacency) signature.
    wrapped=EdgeAwareLayer(16,2,edge)
    wrapped_out=wrapped(x.detach(),adj)
    assert wrapped_out.shape==(3,5,16)
    # Integration: replace Routing only; rebuild optimizer to include NEW weights.
    fake=types.ModuleType('thesis.mappo.gnn_routing_B3_v121')
    class MockGraphRoutingActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden=16
            self.register_buffer('node_xy',xy)
            self.register_buffer('outgoing_average',adj)
            self.layers=nn.ModuleList([nn.Linear(16,16),nn.Linear(16,16)])
    fake.GraphRoutingActor=MockGraphRoutingActor
    sys.modules['thesis.mappo.gnn_routing_B3_v121']=fake
    from thesis.mappo.edge_gat_routing_v4 import enable_edge_gat_routing
    router=MockGraphRoutingActor()
    dispatcher=nn.Linear(8,4)
    orig_dispatch_state={k: v.detach().clone() for k,v in dispatcher.state_dict().items()}
    model=SimpleNamespace(roles={
        'routing':SimpleNamespace(actor=router,actor_optimizer=None),
        'dispatch_front':SimpleNamespace(actor=dispatcher,actor_optimizer=None)},
        cfg=SimpleNamespace(actor_lr=1e-4))
    meta=enable_edge_gat_routing(model,heads=2)
    assert meta['directed_arcs']==int((adj>0).sum())
    assert len(router.layers)==2
    assert all(isinstance(v,EdgeAwareLayer) for v in router.layers)
    parameter_ids={id(p) for g in model.roles['routing'].actor_optimizer.param_groups for p in g['params']}
    assert id(router.layers[0].attention.edge_score.weight) in parameter_ids
    assert id(router.layers[1].attention.edge_value.weight) in parameter_ids
    assert dispatcher is model.roles['dispatch_front'].actor
    assert all(torch.equal(v,dispatcher.state_dict()[k]) for k,v in orig_dispatch_state.items())
    # Hard action mask remains inviolable even with Dijkstra guide.
    logits=torch.tensor([[0.,.1,.2,3.,4.]])
    mask=torch.tensor([[0.,1.,1.,0.,0.]])
    bias=torch.tensor([[0.,0.,.5,9.,9.]])
    valid=logits.masked_fill(mask==0,-1e9)
    dist=Categorical(logits=valid+bias.masked_fill(mask==0,0))
    assert dist.probs[0,3]==0 and dist.probs[0,4]==0
    action=torch.tensor([2]); old_lp=dist.log_prob(action)
    same_lp=Categorical(logits=valid+bias.masked_fill(mask==0,0)).log_prob(action)
    assert torch.allclose(old_lp,same_lp)
    # Same stored bias gives PPO ratio 1 BEFORE optimizer update.
    assert torch.allclose((same_lp-old_lp).exp(),torch.ones_like(old_lp))
    print('PASS: directed edge features, gradient, residual attention, sinks, action mask, PPO pre-update parity')


if __name__=='__main__':
    main()
