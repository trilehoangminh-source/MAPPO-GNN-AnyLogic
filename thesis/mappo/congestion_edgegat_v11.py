"""V11 traffic-aware Residual Edge-GAT for V8 Tobias/AnyLogic MAPPO.

Only Routing actor changes. Each traffic feature is a deterministic function
of the SAME actor observation saved for PPO; never reads live env state in forward.

Dynamic edge channels:
  0 current directed occupancy; 1 discounted predicted shortest-path flow;
  2 upstream waiting demand; 3 downstream stationary load;
  4 signed upstream-minus-downstream pressure; 5 opposing predicted flow.
These are proxies inferred from AGV coordinates, NOT measured physical density,
true queue duration, collision reservation or a deadlock detector.

Architecture adds zero-initialized branches to original V8 q/k/v edge attention
and original edge message. Install AFTER fully loading source V8 source checkpoint.
"""
from __future__ import annotations

import contextvars
import heapq
import math
from types import MethodType

import torch
from torch import nn

_TRAFFIC = contextvars.ContextVar('v11_traffic_features', default=None)
TRAFFIC_NAMES = ('edge_occupancy', 'predicted_flow', 'upstream_waiting',
                 'downstream_load', 'pressure', 'contraflow')


def _all_pairs_next_hop(xy: torch.Tensor, arcs: torch.Tensor) -> torch.Tensor:
    """Directed, static, distance-based next hop for EVERY node and destination."""
    n = int(xy.shape[0])
    coords = xy.detach().cpu().tolist()
    edges = arcs.detach().cpu().tolist()
    neighbors = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if edges[u][v]:
                cost = max(math.dist(coords[u], coords[v]), 1e-6)
                neighbors[u].append((v, cost))
    hops = [[i for _ in range(n)] for i in range(n)]
    for source in range(n):
        dist = [float('inf')] * n
        prev = [-1] * n
        dist[source] = 0.
        queue = [(0., source)]
        while queue:
            cost, u = heapq.heappop(queue)
            if cost > dist[u] + 1e-12:
                continue
            for v, w in neighbors[u]:
                cand = cost + w
                if cand < dist[v] - 1e-12:
                    dist[v], prev[v] = cand, u
                    heapq.heappush(queue, (cand, v))
        for dest in range(n):
            if source == dest or prev[dest] < 0:
                continue
            cur = dest
            for _ in range(n):
                if prev[cur] == source:
                    hops[source][dest] = cur
                    break
                if prev[cur] < 0:
                    break
                cur = prev[cur]
    return torch.tensor(hops, dtype=torch.long, device=xy.device)


class ObservationTrafficFeatures(nn.Module):
    """Traffic computed only from [B,A,F] routing actor observation at each call."""
    def __init__(self, node_xy: torch.Tensor, adjacency: torch.Tensor,
                 horizon: int = 4, decay: float = .75):
        super().__init__()
        if not 1 <= horizon <= 12 or not 0 < decay <= 1:
            raise ValueError('Invalid traffic horizon/decay')
        if node_xy.ndim != 2 or node_xy.shape[1] != 2:
            raise ValueError('node_xy must be [N,2]')
        n = node_xy.shape[0]
        if adjacency.shape != (n,n):
            raise ValueError('Bad graph adjacency shape')
        self.horizon = horizon
        self.decay = decay
        mask = adjacency > 0
        self.register_buffer('node_xy', node_xy.detach().clone())
        self.register_buffer('arc_mask', mask.detach().clone())
        self.register_buffer('next_hop', _all_pairs_next_hop(node_xy, mask))

    def match(self, coords: torch.Tensor):
        delta = (coords.unsqueeze(-2) - self.node_xy).abs().amax(dim=-1)
        ix = delta.argmin(dim=-1)
        matched = delta.gather(-1, ix.unsqueeze(-1)).squeeze(-1) < 1e-3
        return ix, matched

    def forward(self, obs: dict[str, torch.Tensor]):
        agvs = obs['agvs']
        if agvs.ndim != 3 or agvs.shape[-1] < 16:
            raise ValueError('V11 needs original routing AGV observations [B,A,F>=16]')
        if not torch.isfinite(agvs).all():
            raise ValueError('Nonfinite AGV data for traffic features')
        b, a, _ = agvs.shape
        n = self.node_xy.shape[0]
        available = (agvs.abs().sum(-1) > 0) & (agvs[...,0] > 0.5)
        moving = available & (agvs[...,1] > 0.5)
        last, last_ok = self.match(agvs[...,2:4])
        nxt, nxt_ok = self.match(agvs[...,4:6])
        target, target_ok = self.match(agvs[...,6:8])
        # AGV 'next_node' is a waypoint even while traversing last->next.
        # Nonmoving vehicles use next_node, with last_node as fallback.
        start = torch.where(nxt_ok, nxt, last)
        start_ok = nxt_ok | last_ok
        can_route = available & start_ok & target_ok
        # This is just count occupancy of directed last->next traversals,
        # not a claim that the physical lane has a known capacity.
        occupies = moving & last_ok & nxt_ok & (last != nxt) & self.arc_mask[last,nxt]
        occupancy = torch.zeros((b,n*n), device=agvs.device, dtype=agvs.dtype)
        occupancy.scatter_add_(1, last*n+nxt, occupies.to(agvs.dtype))
        occupancy = occupancy.reshape(b,n,n)
        # Idle AGV near a downstream node is an approximate load, not
        # necessarily a real queue (it might be undergoing processing).
        waiting = can_route & ~moving
        idle_node = torch.zeros((b,n), device=agvs.device, dtype=agvs.dtype)
        idle_node.scatter_add_(1,start,waiting.to(agvs.dtype))
        downstream = idle_node.unsqueeze(1).expand(b,n,n)
        # Predict traffic from current observable targets using static Dijkstra.
        # NO future simulator state, hidden reservation, or future action used.
        flow = torch.zeros((b,n*n), device=agvs.device, dtype=agvs.dtype)
        upstream = torch.zeros_like(flow)
        at = start
        active = can_route
        for step in range(self.horizon):
            hop = self.next_hop[at,target]
            viable = active & (at != target) & (hop != at) & self.arc_mask[at,hop]
            indices = at*n + hop
            flow.scatter_add_(1, indices, viable.to(agvs.dtype) * (self.decay ** step))
            if step == 0:
                upstream.scatter_add_(1, indices, (viable & waiting).to(agvs.dtype))
            at = torch.where(viable, hop, at)
            active = active & viable
        flow = flow.reshape(b,n,n)
        upstream = upstream.reshape(b,n,n)
        count = float(max(a,1))
        downstream = downstream * self.arc_mask.unsqueeze(0)
        opposing = flow * flow.transpose(1,2) * (self.arc_mask & self.arc_mask.T).unsqueeze(0)
        # Keep signed pressure channel: high upstream demand vs downstream load.
        pressure = upstream - downstream
        result = torch.stack((occupancy/count, flow/count, upstream/count,
                              downstream/count, pressure/count, opposing/(count*count)), dim=-1)
        result = result * self.arc_mask.unsqueeze(0).unsqueeze(-1)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite V11 traffic tensor')
        return result


class CongestionResidualEdgeGAT(nn.Module):
    """Exact V8 modules + learnable traffic logits/message with zero initial effect."""
    def __init__(self, pretrained):
        super().__init__()
        # Keep the original MODULE INSTANCES and parameter identities.
        for name in ('q','k','v','edge_score','edge_value','output','norm'):
            setattr(self,name,getattr(pretrained,name))
        self.hidden = int(pretrained.hidden)
        self.heads = int(pretrained.heads)
        self.head_dim = int(pretrained.head_dim)
        self.traffic_score = nn.Linear(len(TRAFFIC_NAMES), self.heads, bias=False)
        self.traffic_value = nn.Linear(len(TRAFFIC_NAMES), self.hidden, bias=False)
        nn.init.zeros_(self.traffic_score.weight)
        nn.init.zeros_(self.traffic_value.weight)

    def forward(self,node,outgoing_average,edge_features):
        traffic = _TRAFFIC.get()
        if traffic is None:
            raise RuntimeError('V11 traffic context missing; call routing actor logits/dist')
        if node.ndim != 3 or edge_features.ndim != 3:
            raise ValueError('V11 invalid node/static edge shape')
        batch,n,hidden = node.shape
        if (hidden != self.hidden or outgoing_average.shape != (n,n)
                or edge_features.shape != (n,n,3)
                or traffic.shape != (batch,n,n,len(TRAFFIC_NAMES))):
            raise ValueError('V11 incompatible traffic or graph shapes')
        valid = (outgoing_average > 0) | torch.eye(n,device=node.device,dtype=torch.bool)
        def reshape(t):
            return t.reshape(batch,n,self.heads,self.head_dim).transpose(1,2)
        q,k,v = (reshape(layer(node)) for layer in (self.q,self.k,self.v))
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_score(edge_features).permute(2,0,1).unsqueeze(0)
        dyn_scores = self.traffic_score(traffic).permute(0,3,1,2)
        scores = scores + dyn_scores
        scores = scores.masked_fill(~valid[None,None],torch.finfo(scores.dtype).min)
        prob = torch.softmax(scores,dim=-1)
        attended_nodes = prob @ v
        edge_msg = self.edge_value(edge_features).reshape(n,n,self.heads,self.head_dim).permute(2,0,1,3)
        attended_edges = torch.einsum('bkij,kijd->bkid',prob,edge_msg)
        # Compute aggregated traffic in F=6 space BEFORE the dense projection;
        # avoids enormous B*N*N*hidden tensors and saves GPU memory.
        weighted_traffic = torch.einsum('bkij,bijf->bkif',prob,traffic)
        traffic_w = self.traffic_value.weight.reshape(self.heads,self.head_dim,len(TRAFFIC_NAMES))
        traffic_msg = torch.einsum('bkif,kdf->bkid',weighted_traffic,traffic_w)
        combined = (attended_nodes+attended_edges+traffic_msg).transpose(1,2).contiguous().reshape(batch,n,hidden)
        return torch.tanh(self.norm(node+self.output(combined)))


def install_v11(model, horizon:int=4, decay:float=.75):
    """Upgrade in-place AFTER model.load(V8 source); preserve all V8 params/optim.

    On RESUME recreate this upgrade first, then model.load(V11 checkpoint).
    """
    from thesis.mappo.edge_gat_routing_v4 import EdgeAwareLayer, ResidualEdgeGAT
    actor = model.roles['routing'].actor
    if hasattr(actor,'v11_traffic'):
        raise RuntimeError('V11 already installed')
    layers = actor.layers
    if not layers or not all(isinstance(layer,EdgeAwareLayer) for layer in layers):
        raise TypeError('Expected V8 routing EdgeAwareLayer stack')
    added = []
    for layer in layers:
        old = layer.attention
        if not isinstance(old,ResidualEdgeGAT):
            raise TypeError('Expected V8 ResidualEdgeGAT attention')
        layer.attention = new = CongestionResidualEdgeGAT(old).to(actor.node_xy.device)
        added.extend([new.traffic_score.weight,new.traffic_value.weight])
    actor.v11_traffic = ObservationTrafficFeatures(actor.node_xy, actor.outgoing_average,
                                                   horizon=horizon,decay=decay).to(actor.node_xy.device)
    # Routing .dist may have already been wrapped by V8's Dijkstra guide;
    # calling original logits preserves that wrapper and hard action mask.
    base_logits = actor.logits
    def traffic_logits(self,obs):
        traffic = self.v11_traffic(obs)
        token = _TRAFFIC.set(traffic)
        try:
            return base_logits(obs)
        finally:
            _TRAFFIC.reset(token)
    actor.logits = MethodType(traffic_logits,actor)
    opt = model.roles['routing'].actor_optimizer
    opt.add_param_group({'params':added})  # retains V8 moments and adds new params
    return {'channels':TRAFFIC_NAMES,'horizon':horizon,'decay':decay,
            'traffic_weights_zero':all(torch.count_nonzero(p).item()==0 for p in added),
            'new_parameter_tensors':len(added)}


def synthetic_parity_probe(model, atol=2e-5):
    """Run BEFORE and AFTER installing V11 by calling this with original actor.

    Returned observation matches feature layout, not an AnyLogic simulation.
    """
    actor = model.roles['routing'].actor
    device = actor.node_xy.device
    feature_dim = actor.entity_encoder.main[0].in_features
    n = actor.node_xy.shape[0]
    agvs = torch.zeros((1,20,feature_dim),device=device)
    agvs[0,0,0] = 1
    agvs[0,0,2:4] = actor.node_xy[0]
    agvs[0,0,4:6] = actor.node_xy[0]
    agvs[0,0,6:8] = actor.node_xy[min(1,n-1)]
    for j in range(4):
        agvs[0,0,8+2*j:10+2*j] = actor.node_xy[min(j,n-1)]
    obs = {'agvs':agvs,'stat':torch.zeros((1,1,feature_dim),device=device),
           'action_mask':torch.ones((1,5),device=device)}
    with torch.no_grad():
        actor.eval()
        return obs,actor.logits(obs).detach().clone()
