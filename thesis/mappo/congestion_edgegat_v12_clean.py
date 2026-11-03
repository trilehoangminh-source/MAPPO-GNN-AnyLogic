"""V12 fully no shortest-path: observation-only edge traffic for Routing GAT.

NO Dijkstra, NO all-pairs shortest paths, NO target-based path forecasts,
NO future state, NO old guidance. Source channels (V11 position) are replaced
in-place for checkpoint weight/optimizer compatibility:
  0 occupancy: agents moving on a directed edge
  1 intent: agents' observed last->next waypoint, moving OR waiting
  2 upstream_waiting: stationary agents at upstream node
  3 downstream_load: stationary agents at downstream node
  4 pressure: upstream_waiting - downstream_load
  5 contraflow: product of opposite-direction *observed* intentions
The six feature units are normalized by fleet observation size; they are
proxies, not physical congestion measurements or safety guarantees.

V11 -> V12 weight migration: keep all trainable weights and optimizer moments;
DROP v11_traffic.next_hop buffer ONLY (it is historical static planner data).
Do NOT change the original V11 source file; this V12 module is independent.
"""
from __future__ import annotations
import contextvars
import math
from types import MethodType
import torch
from torch import nn

_TRAFFIC = contextvars.ContextVar('v12_traffic_features', default=None)
TRAFFIC_NAMES = ('edge_occupancy', 'observed_edge_intent', 'upstream_waiting',
                 'downstream_load', 'pressure', 'observed_contraflow')


class ObservationOnlyTraffic(nn.Module):
    def __init__(self, node_xy: torch.Tensor, adjacency: torch.Tensor):
        super().__init__()
        if node_xy.ndim != 2 or node_xy.shape[-1] != 2:
            raise ValueError('node_xy must be [N,2]')
        n = int(node_xy.shape[0])
        if adjacency.shape != (n, n):
            raise ValueError('adjacency must be [N,N]')
        self.register_buffer('node_xy', node_xy.detach().clone())
        self.register_buffer('arc_mask', (adjacency > 0).detach().clone())
        # Intentionally NO next_hop, graph-distance table, route forecast or cache.

    def match(self, coords: torch.Tensor):
        delta = (coords.unsqueeze(-2) - self.node_xy).abs().amax(dim=-1)
        ix = delta.argmin(dim=-1)
        okay = delta.gather(-1, ix.unsqueeze(-1)).squeeze(-1) < 1e-3
        return ix, okay

    def forward(self, obs: dict[str, torch.Tensor]):
        agvs = obs['agvs']
        if agvs.ndim != 3 or agvs.shape[-1] < 16:
            raise ValueError('Requires actor-visible AGV observations [B,A,F>=16]')
        if not torch.isfinite(agvs).all():
            raise ValueError('Nonfinite AGV observations')
        b, a, _ = agvs.shape
        n = int(self.node_xy.shape[0])
        # These columns are exactly the ORIGINAL V11 observation layout.
        present = (agvs.abs().sum(-1) > 0) & (agvs[..., 0] > .5)
        moving = present & (agvs[..., 1] > .5)
        last, last_ok = self.match(agvs[..., 2:4])
        nxt, nxt_ok = self.match(agvs[..., 4:6])
        # All features are based on observed current positions and next waypoint.
        # The target coords agvs[...,6:8] are NEVER used for any path estimate.
        directed = present & last_ok & nxt_ok & (last != nxt) & self.arc_mask[last, nxt]
        occupies = directed & moving
        count = float(max(a, 1))
        idx = last * n + nxt
        occupancy = torch.zeros((b, n*n), dtype=agvs.dtype, device=agvs.device)
        intent = torch.zeros_like(occupancy)
        occupancy.scatter_add_(1, idx, occupies.to(agvs.dtype))
        intent.scatter_add_(1, idx, directed.to(agvs.dtype))
        occupancy = occupancy.reshape(b, n, n)
        intent = intent.reshape(b, n, n)
        # Approximate stationary load at the current node. If last is unmapped,
        # fall back to next. Do not infer a route for an idle agent.
        parked_node = torch.where(last_ok, last, nxt)
        stationary = present & ~moving & (last_ok | nxt_ok)
        stopped = torch.zeros((b, n), dtype=agvs.dtype, device=agvs.device)
        stopped.scatter_add_(1, parked_node, stationary.to(agvs.dtype))
        upstream = stopped.unsqueeze(2).expand(b, n, n)
        downstream = stopped.unsqueeze(1).expand(b, n, n)
        pressure = upstream - downstream
        both = (self.arc_mask & self.arc_mask.T).unsqueeze(0)
        opposing = intent * intent.transpose(1, 2) * both
        result = torch.stack((occupancy / count, intent / count,
                              upstream / count, downstream / count,
                              pressure / count, opposing / (count * count)), dim=-1)
        result = result * self.arc_mask.unsqueeze(0).unsqueeze(-1)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite V12 traffic tensor')
        return result


class CongestionResidualEdgeGAT(nn.Module):
    """Same trainable module names and shapes as V11: strict weight compatibility."""
    def __init__(self, pretrained):
        super().__init__()
        for name in ('q', 'k', 'v', 'edge_score', 'edge_value', 'output', 'norm'):
            setattr(self, name, getattr(pretrained, name))
        self.hidden = int(pretrained.hidden)
        self.heads = int(pretrained.heads)
        self.head_dim = int(pretrained.head_dim)
        self.traffic_score = nn.Linear(len(TRAFFIC_NAMES), self.heads, bias=False)
        self.traffic_value = nn.Linear(len(TRAFFIC_NAMES), self.hidden, bias=False)
        nn.init.zeros_(self.traffic_score.weight)
        nn.init.zeros_(self.traffic_value.weight)

    def forward(self, node, outgoing_average, edge_features):
        traffic = _TRAFFIC.get()
        if traffic is None:
            raise RuntimeError('V12 traffic context missing')
        if node.ndim != 3 or edge_features.ndim != 3:
            raise ValueError('Invalid edge GAT input shapes')
        batch, n, hidden = node.shape
        if (hidden != self.hidden or outgoing_average.shape != (n, n) or
                edge_features.shape != (n, n, 3) or
                traffic.shape != (batch, n, n, len(TRAFFIC_NAMES))):
            raise ValueError('V12 traffic tensor shape mismatch')
        valid = (outgoing_average > 0) | torch.eye(n, device=node.device, dtype=torch.bool)
        def reshape(t):
            return t.reshape(batch, n, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = (reshape(layer(node)) for layer in (self.q, self.k, self.v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_score(edge_features).permute(2, 0, 1).unsqueeze(0)
        scores = scores + self.traffic_score(traffic).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~valid[None, None], torch.finfo(scores.dtype).min)
        prob = torch.softmax(scores, dim=-1)
        attended_nodes = prob @ v
        edge_msg = self.edge_value(edge_features).reshape(n, n, self.heads, self.head_dim).permute(2, 0, 1, 3)
        attended_edges = torch.einsum('bkij,kijd->bkid', prob, edge_msg)
        weighted_traffic = torch.einsum('bkij,bijf->bkif', prob, traffic)
        traffic_w = self.traffic_value.weight.reshape(self.heads, self.head_dim, len(TRAFFIC_NAMES))
        traffic_msg = torch.einsum('bkif,kdf->bkid', weighted_traffic, traffic_w)
        combined = (attended_nodes + attended_edges + traffic_msg).transpose(1, 2).contiguous().reshape(batch, n, hidden)
        return torch.tanh(self.norm(node + self.output(combined)))


def install_v12(model):
    """Replace V8 traffic-free attention with V12, retaining optimizer identities.

    NO shortest path planner is built. All Routing/Dispatch weights remain trainable.
    """
    from thesis.mappo.edge_gat_routing_v4 import EdgeAwareLayer, ResidualEdgeGAT
    from thesis.mappo.algorithm import MAPPO
    actor = model.roles['routing'].actor
    if hasattr(actor, 'v11_traffic'):
        raise RuntimeError('Traffic already installed')
    if not actor.layers or not all(isinstance(x, EdgeAwareLayer) for x in actor.layers):
        raise TypeError('V8 Edge-GAT expected')
    added = []
    for layer in actor.layers:
        old = layer.attention
        if not isinstance(old, ResidualEdgeGAT):
            raise TypeError('Expected V8 attention for V12 installation')
        layer.attention = new = CongestionResidualEdgeGAT(old).to(actor.node_xy.device)
        added.extend([new.traffic_score.weight, new.traffic_value.weight])
    # Historical attribute name is retained solely for strict V11 checkpoint keys.
    actor.v11_traffic = ObservationOnlyTraffic(actor.node_xy, actor.outgoing_average).to(actor.node_xy.device)
    original_logits = actor.logits
    def traffic_logits(self, obs):
        token = _TRAFFIC.set(self.v11_traffic(obs))
        try:
            return original_logits(obs)
        finally:
            _TRAFFIC.reset(token)
    actor.logits = MethodType(traffic_logits, actor)
    model.roles['routing'].actor_optimizer.add_param_group({'params': added})
    # Remove V8/GuidedMAPPO additive-prior wrapper instead of feeding zero bias.
    actor.dist = MethodType(type(actor).dist, actor)
    actor._b3_guide_dist = False
    model.act = MethodType(MAPPO.act, model)
    if any(not p.requires_grad for role in model.roles.values()
           for net in (role.actor, role.critic) for p in net.parameters()):
        raise RuntimeError('A Routing or Dispatch weight is frozen')
    return {'channels': TRAFFIC_NAMES, 'source_shortest_path_flow': False,
            'guidance_removed': True, 'trainable_traffic_tensors': len(added)}
