"""Opt-in graph attention Routing encoder, inspired by Tobias' routing GAT.

Self-contained torch implementation: does NOT require PyG or modify Dispatching.
Directed adjacency from the installed B3-GNN GraphRoutingActor is respected.
This is an independently written adapter, not copied from uoe-agents sources.
"""
from __future__ import annotations

import torch
from torch import nn


class DirectedGraphAttention(nn.Module):
    """Multi-head attention to outgoing successors, with self-attention.

    An edge (u,v) lets u attend to v. The diagonal is included so isolated
    nodes remain well-defined. Topology uses graph buffers, not global state.
    """

    def __init__(self, hidden: int, heads: int = 2):
        super().__init__()
        if hidden < 4 or heads < 1 or hidden % heads:
            raise ValueError("hidden must be >=4 and divisible by heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, node: torch.Tensor, outgoing_average: torch.Tensor):
        if node.ndim != 3 or outgoing_average.ndim != 2:
            raise ValueError("expected [B,N,H] nodes and [N,N] adjacency")
        b, n, h = node.shape
        if h != self.hidden or outgoing_average.shape != (n, n):
            raise ValueError("attention graph dimension mismatch")
        reshape = lambda x: x.reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = (reshape(f(node)) for f in (self.q, self.k, self.v))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        valid = outgoing_average.to(node.device) > 0
        valid = valid | torch.eye(n, device=node.device, dtype=torch.bool)
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().reshape(b, n, h)
        return torch.tanh(self.norm(node + self.out(attended)))


def enable_gat_routing(model, heads: int = 2):
    """Switch ONLY the existing B3 Routing actor's MP layers to GAT.

    Call AFTER upgrade_guided_mappo_to_gnn() and BEFORE first rollout.
    Important: rebuild optimizer because new module has new parameters.
    Existing guidance wrapper calls actor.dist -> actor.logits dynamically.
    """
    from thesis.mappo.gnn_routing_B3_v121 import GraphRoutingActor

    if 'routing' not in model.roles:
        raise RuntimeError('Missing routing role')
    role = model.roles['routing']
    actor = role.actor
    if not isinstance(actor, GraphRoutingActor):
        raise TypeError('Expected B3 GraphRoutingActor; do not replace Dispatching actor')
    if getattr(actor, '_uoe_gat_enabled', False):
        raise RuntimeError('GAT already enabled')
    count = len(actor.layers)
    actor.layers = nn.ModuleList(DirectedGraphAttention(actor.hidden, heads) for _ in range(count))
    role.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=model.cfg.actor_lr)
    actor._uoe_gat_enabled = True
    return dict(encoder='directed_gat', graph_layers=count, heads=heads)
