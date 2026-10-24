"""B3-GNN: directed graph message passing Routing Actor for Tobias/AnyLogic.

Additive experiment module.  Does NOT modify B2, Java, collector, or PPO.
The graph is a frozen structural map from HiveContext.nodes and .paths;
all per-decision features come exclusively from the actor observation.
Use with existing GuidedMAPPO + GuidedEventCollector so Dijkstra logit bias
is applied identically in rollout and PPO updates.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _coord(x):
    return [float(x[0]), float(x[1])]


@dataclass(frozen=True)
class GraphSpec:
    ids: tuple
    xy: tuple
    arcs: tuple
    fingerprint: str

    @classmethod
    def from_hive(cls, hive):
        nodes = getattr(hive, 'nodes', None)
        paths = getattr(hive, 'paths', None)
        if not nodes or not paths:
            raise RuntimeError('B3-GNN needs LIVE HiveContext.nodes and paths after env.reset().')
        ids = tuple(sorted(int(k) for k in nodes))
        idx = {v: i for i, v in enumerate(ids)}
        xy = tuple(tuple(_coord(nodes[i])) for i in ids)
        if not np.isfinite(np.asarray(xy, dtype=float)).all():
            raise ValueError('Nonfinite Hive node coordinates')
        if len(set(xy)) != len(xy):
            raise ValueError('Ambiguous Hive node coordinates; cannot match agent observation')
        arcs = tuple(sorted({(idx[int(a)], idx[int(b)]) for a, b in paths
                             if int(a) in idx and int(b) in idx and int(a) != int(b)}))
        if not arcs:
            raise ValueError('Hive has no usable directed arcs')
        canonical = json.dumps(dict(ids=ids, xy=xy, arcs=arcs), separators=(',', ':'))
        return cls(ids, xy, arcs, hashlib.sha256(canonical.encode('utf8')).hexdigest())

    @property
    def n(self):
        return len(self.ids)


class DirectedMP(nn.Module):
    """Successor-to-predecessor message passing along DIRECTED outgoing arcs."""
    def __init__(self, hidden):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())

    def forward(self, x, outgoing_average):
        neigh = torch.matmul(outgoing_average, x)
        return self.update(torch.cat((x, neigh), dim=-1))


class GraphRoutingActor(nn.Module):
    """GNN policy for 5 actions (Stop + four Tobias neighbor slots).

    No torch_geometric dependency. Fixed topology is included as registered
    buffers/checkpoint. Dynamic node features use ONLY 'agvs' actor input.
    Station/other-agent features are pooled from the SAME actor observation.
    Candidate node embeddings give action-specific directional logits.
    """
    action_dim = 5

    def __init__(self, obs_space, graph: GraphSpec, hidden=32, layers=2):
        super().__init__()
        if hidden < 4 or layers < 1:
            raise ValueError('Require hidden >= 4 and at least one GNN layer')
        from thesis.mappo.networks import EntityEncoder, masked_logits
        dim = int(obs_space['agvs'].shape[-1])
        if dim < 16:
            raise ValueError('Expected original Tobias AGV coordinates columns 4:16')
        self.n = graph.n
        self.hidden = int(hidden)
        xy = torch.as_tensor(graph.xy, dtype=torch.float32)
        self.register_buffer('node_xy', xy)
        span = (xy.max(0).values - xy.min(0).values).max().clamp(min=1.0)
        self.register_buffer('scale', span)
        # [receiver, sender]: each node receives messages from reachable successors
        adjacency = torch.zeros((graph.n, graph.n), dtype=torch.float32)
        for src, dst in graph.arcs:
            adjacency[src, dst] = 1.
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp(min=1.)
        self.register_buffer('outgoing_average', adjacency)
        self.entity_encoder = EntityEncoder(dim, hidden)
        self.context = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.Tanh())
        # normalized x,y, position relative to ego and goal, distances, occupancy,
        # is_ego, is_target  => 2+2+2+1+1+1+1+1 = 11
        self.node_in = nn.Sequential(nn.Linear(11, hidden), nn.Tanh())
        self.layers = nn.ModuleList(DirectedMP(hidden) for _ in range(layers))
        self.fallback = nn.Sequential(nn.Linear(4, hidden), nn.Tanh())
        self.stop = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.move = nn.Sequential(nn.Linear(4 * hidden + 3, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def _match(self, coords, tolerance=1e-3):
        """Closest exact graph node, plus valid-match flag; supports [B,K,2]."""
        distance = (coords.unsqueeze(-2) - self.node_xy).abs().amax(dim=-1)
        nearest = distance.argmin(-1)
        return nearest, distance.gather(-1, nearest.unsqueeze(-1)).squeeze(-1) < tolerance

    def logits(self, obs):
        agvs, stat = obs['agvs'], obs['stat']
        if agvs.ndim != 3 or stat.ndim != 3 or agvs.shape[-1] < 16:
            raise ValueError('B3-GNN expects batched Tobias agvs/stat feature matrices')
        if not torch.isfinite(agvs).all() or not torch.isfinite(stat).all():
            raise ValueError('Actor observation contains nonfinite features')
        batch = agvs.shape[0]
        ego = agvs[:, 0]
        pos, target = ego[:, 4:6], ego[:, 6:8]
        neighbors = ego[:, 8:16].reshape(batch, 4, 2)
        loc = self.node_xy.unsqueeze(0).expand(batch, -1, -1)
        pos_delta = (loc - pos.unsqueeze(1)) / self.scale
        tar_delta = (loc - target.unsqueeze(1)) / self.scale
        # Actor-visible AGV positions; padded zero entities must NOT imply a vehicle at (0,0).
        existing = agvs.abs().sum(-1) > 0
        dist_to_agvs = (loc.unsqueeze(2) - agvs[:, None, :, 4:6]).abs().amax(-1)
        occupancy = ((dist_to_agvs < 1e-3) & existing.unsqueeze(1)).sum(-1).to(agvs.dtype)
        ego_flag = (pos_delta.abs().amax(-1) < 1e-3).to(agvs.dtype)
        tar_flag = (tar_delta.abs().amax(-1) < 1e-3).to(agvs.dtype)
        features = torch.cat((loc/self.scale, pos_delta, tar_delta,
                              torch.linalg.vector_norm(pos_delta, dim=-1, keepdim=True),
                              torch.linalg.vector_norm(tar_delta, dim=-1, keepdim=True),
                              occupancy.unsqueeze(-1), ego_flag.unsqueeze(-1),
                              tar_flag.unsqueeze(-1)), dim=-1)
        h = self.node_in(features)
        for layer in self.layers:
            h = layer(h, self.outgoing_average)
        pos_i, pos_valid = self._match(pos.unsqueeze(1))
        tar_i, tar_valid = self._match(target.unsqueeze(1))
        mov_i, mov_valid = self._match(neighbors)
        b = torch.arange(batch, device=agvs.device)
        # For unmapped geometry, preserve differentiability via coordinate fallback.
        pos_emb = torch.where(pos_valid.unsqueeze(-1), h[b, pos_i[:, 0]].unsqueeze(1),
                              self.fallback(torch.cat((pos/self.scale, (pos-target)/self.scale), -1)).unsqueeze(1))[:, 0]
        tar_emb = torch.where(tar_valid.unsqueeze(-1), h[b, tar_i[:, 0]].unsqueeze(1),
                              self.fallback(torch.cat((target/self.scale, (target-pos)/self.scale), -1)).unsqueeze(1))[:, 0]
        sampled = h[b.unsqueeze(-1), mov_i]
        mov_fallback = self.fallback(torch.cat(((neighbors-pos.unsqueeze(1))/self.scale,
                                                (neighbors-target.unsqueeze(1))/self.scale), -1))
        successor = torch.where(mov_valid.unsqueeze(-1), sampled, mov_fallback)
        context = self.context(self.entity_encoder(obs))
        stops = self.stop(torch.cat((context, pos_emb, tar_emb), -1))
        block = torch.cat((context.unsqueeze(1).expand(-1,4,-1),
                           pos_emb.unsqueeze(1).expand(-1,4,-1),
                           tar_emb.unsqueeze(1).expand(-1,4,-1), successor,
                           (neighbors-pos.unsqueeze(1))/self.scale,
                           torch.linalg.vector_norm((neighbors-target.unsqueeze(1))/self.scale,
                                                    dim=-1, keepdim=True)), -1)
        moves = self.move(block).squeeze(-1)
        return torch.cat((stops, moves), dim=-1)

    def dist(self, obs):
        from thesis.mappo.networks import masked_logits
        return Categorical(logits=masked_logits(self.logits(obs), obs['action_mask']))


def upgrade_guided_mappo_to_gnn(model, hive, hidden=32, layers=2):
    """Replace ONLY Routing actor. Must be called before the first rollout or load."""
    from thesis.mappo.routing_guidance_v121 import enable_routing_distribution_prior
    from thesis.mappo.networks import Actor
    spec = GraphSpec.from_hive(hive)
    role = model.roles['routing']
    old = role.actor
    # Use the existing pooled encoder observation space dimensions, not env globals.
    # EntityEncoder's dim is extracted from the old pooled encoder's first Linear.
    if not hasattr(old, 'encoder') or not hasattr(old.encoder, 'main'):
        raise RuntimeError('Expected baseline pooled routing actor; use --routing-backbone pooled during construction')
    feature_dim = old.encoder.main[0].in_features
    from types import SimpleNamespace
    observation_shapes = {'agvs': SimpleNamespace(shape=(1,feature_dim))}
    actor = GraphRoutingActor(observation_shapes, spec, hidden=hidden, layers=layers).to(model.device)
    role.actor = actor
    # Required: old optimizer otherwise points at OLD pooled parameters.
    role.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=model.cfg.actor_lr)
    # Existing B3 soft-Dijkstra prior is now wrapped around the new GNN dist.
    enable_routing_distribution_prior(model)
    model.gnn_graph_fingerprint = spec.fingerprint
    model.gnn_graph_nodes, model.gnn_graph_edges = spec.n, len(spec.arcs)
    model.gnn_layers = layers
    return spec
