"""Edge-aware residual directed graph attention for Tobias AGV Routing ONLY.

Adaptation of the ideas in Lei et al. (2022), "Solve routing problems with a
residual edge-graph attention neural network". Original independent code,
not copied from the cited GitHub repository. Only the Routing actor is replaced.

Edge features use graph coordinate units normalized by graph span, NOT meters.
The action masking and Dijkstra distribution wrapper are unchanged.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class ResidualEdgeGAT(nn.Module):
    """Outgoing-directed multihead edge attention with residual node update.

    Node u attends to node v only if the graph contains directed arc u->v;
    self-attention handles sinks. Besides the destination node embedding,
    edge value/direction participates directly in the attention and message.
    """

    def __init__(self, hidden: int, heads: int = 2):
        super().__init__()
        if hidden < 4 or heads < 1 or hidden % heads:
            raise ValueError('hidden >= 4, heads >= 1, hidden divisible by heads')
        self.hidden, self.heads = int(hidden), int(heads)
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.edge_score = nn.Linear(3, heads, bias=False)
        self.edge_value = nn.Linear(3, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, node: torch.Tensor, outgoing_average: torch.Tensor,
                edge_features: torch.Tensor):
        if node.ndim != 3 or outgoing_average.ndim != 2 or edge_features.ndim != 3:
            raise ValueError('Expected [B,N,H], [N,N] adjacency and [N,N,3] edges')
        batch, n, hidden = node.shape
        if hidden != self.hidden or outgoing_average.shape != (n,n) or edge_features.shape != (n,n,3):
            raise ValueError('ResidualEdgeGAT graph dimension mismatch')
        if not torch.isfinite(node).all() or not torch.isfinite(edge_features).all():
            raise ValueError('Nonfinite node or edge features')
        valid = outgoing_average > 0
        valid = valid | torch.eye(n, device=node.device, dtype=torch.bool)
        def reshape(t):
            return t.reshape(batch, n, self.heads, self.head_dim).transpose(1,2)
        q, k, v = (reshape(layer(node)) for layer in (self.q, self.k, self.v))
        # Edge score: [N,N,K] -> [1,K,N,N]
        score = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        score = score + self.edge_score(edge_features).permute(2,0,1).unsqueeze(0)
        score = score.masked_fill(~valid[None,None], torch.finfo(score.dtype).min)
        prob = torch.softmax(score, dim=-1)
        # Include weighted directed edge message: [N,N,K,D] and [B,K,N,N].
        edge_message = self.edge_value(edge_features).reshape(n,n,self.heads,self.head_dim).permute(2,0,1,3)
        attended_nodes = prob @ v
        attended_edges = torch.einsum('bkij,kijd->bkid', prob, edge_message)
        combined = (attended_nodes + attended_edges).transpose(1,2).contiguous().reshape(batch,n,hidden)
        return torch.tanh(self.norm(node + self.output(combined)))


class EdgeAwareLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, edge_features: torch.Tensor):
        super().__init__()
        self.attention = ResidualEdgeGAT(hidden, heads)
        self.register_buffer('edge_features', edge_features.clone().detach())

    def forward(self, node, outgoing_average):
        return self.attention(node, outgoing_average, self.edge_features)


def compute_static_edge_features(node_xy: torch.Tensor, outgoing_average: torch.Tensor) -> torch.Tensor:
    """Feature 0: relative length. Feature 1/2: directed normalized dx,dy.

    Edge geometry comes from LIVE HiveContext graph, never inferred by rounding
    or by assuming coordinate units are meters. Non-arcs have zero values.
    """
    if node_xy.ndim != 2 or node_xy.shape[-1] != 2:
        raise ValueError('node_xy must be [N,2]')
    n = node_xy.shape[0]
    if outgoing_average.shape != (n,n):
        raise ValueError('adjacency does not match node coordinates')
    xy = node_xy.float()
    if not torch.isfinite(xy).all():
        raise ValueError('Nonfinite coordinates')
    span = (xy.max(0).values - xy.min(0).values).max().clamp(min=1e-6)
    delta = (xy[None,:,:] - xy[:,None,:]) / span   # delta[u,v] = v - u
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    features = torch.cat((length, delta), dim=-1)
    return features * (outgoing_average > 0).unsqueeze(-1)


def enable_edge_gat_routing(model, heads: int = 2):
    """Replace directed MP layers of Routing ONLY, then rebuild optimizer.

    MUST be called after B3 build_gnn_model but BEFORE loading checkpoint,
    starting the collector, or installing the RoutingGraphAudit.
    """
    from thesis.mappo.gnn_routing_B3_v121 import GraphRoutingActor
    roles = model.roles
    if 'routing' not in roles or not isinstance(roles['routing'].actor, GraphRoutingActor):
        raise RuntimeError('Expected an installed B3 GraphRoutingActor for routing')
    actor = roles['routing'].actor
    if getattr(actor, '_edge_v4', False):
        raise RuntimeError('Residual Edge-GAT already installed')
    for role_name, role in roles.items():
        if role_name != 'routing' and isinstance(role.actor, GraphRoutingActor):
            raise RuntimeError('Dispatching actor was unexpectedly converted to GNN')
    edges = compute_static_edge_features(actor.node_xy, actor.outgoing_average)
    old_count = len(actor.layers)
    actor.layers = nn.ModuleList([
        EdgeAwareLayer(actor.hidden, heads, edges) for _ in range(old_count)
    ]).to(actor.node_xy.device)
    roles['routing'].actor_optimizer = torch.optim.Adam(actor.parameters(), lr=model.cfg.actor_lr)
    actor._edge_v4 = True
    return dict(routing_encoder='residual_edge_gat', graph_layers=old_count,
                heads=heads, feature_order=['relative_edge_length', 'dx', 'dy'],
                edge_units='original graph coordinate units / graph span',
                directed_arcs=int((actor.outgoing_average>0).sum().item()))
