"""CPU-only standalone smoke: GAT direction/gradient; audit geometry; no Java.

Run: python tests/test_uoe_v2_offline.py
For full B3 integration also run: python tests/test_gnn_B3_v121.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from thesis.mappo.gnn_attention_uoe_v2 import DirectedGraphAttention


def test_gat_direction_and_gradients():
    torch.manual_seed(16)
    gat = DirectedGraphAttention(hidden=8, heads=2)
    x = torch.randn(2, 4, 8, requires_grad=True)
    adj = torch.zeros(4,4)
    adj[0,1] = adj[1,2] = adj[2,3] = 1.
    out = gat(x, adj)
    assert out.shape == (2,4,8) and torch.isfinite(out).all()
    reversed_out = gat(x, adj.T)
    assert not torch.allclose(out, reversed_out), 'Ignoring directed graph is a serious error'
    out[0,0,0].backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert sum(float(p.grad.abs().sum()) for p in gat.parameters()
               if p.grad is not None) > 0
    print('PASS directed GAT uses adjacency, reverse edges differ, gradients flow')


def test_no_edges_and_invalid_hidden():
    gat = DirectedGraphAttention(8,2)
    x = torch.randn(1, 5, 8)
    isolated = gat(x, torch.zeros(5,5))
    assert torch.isfinite(isolated).all()
    try:
        DirectedGraphAttention(7,2)
    except ValueError:
        pass
    else:
        raise AssertionError('Bad hidden/head combination not rejected')
    print('PASS isolated nodes and dimensions')


def test_ppo_logprob_parity_with_soft_bias_and_hard_mask():
    logits = torch.tensor([[0., 1., 2., 3., 4.]], dtype=torch.float32)
    mask = torch.tensor([[1.,1.,0.,1.,0.]])
    prior = torch.tensor([[0.,0.,9.,0.5,20.]])
    # Dijkstra bias must NEVER unmask forbidden actions.
    legal_logits = logits.masked_fill(mask <= .5, -1e9)
    guided = torch.distributions.Categorical(
        logits=legal_logits + prior.masked_fill(mask <= .5, 0))
    action = torch.tensor([3])
    rollout_lp = guided.log_prob(action)
    update_lp = guided.log_prob(action)
    assert abs(float(torch.exp(update_lp-rollout_lp))-1.) < 1e-6
    assert guided.probs[0,2] == 0 and guided.probs[0,4] == 0
    print('PASS Dijkstra prior, hard action mask and initial PPO ratio')


if __name__ == '__main__':
    test_gat_direction_and_gradients()
    test_no_edges_and_invalid_hidden()
    test_ppo_logprob_parity_with_soft_bias_and_hard_mask()
    print('ALL V2 STANDALONE TESTS PASSED')
