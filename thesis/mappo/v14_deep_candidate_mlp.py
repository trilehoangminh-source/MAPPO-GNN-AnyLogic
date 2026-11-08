"""V14: independent candidate-aware MLP Dispatch actors and global MLP critics.

No attention/Transformer or pretrained Dispatch MLP is used. Compatible with
MAPPO RolePolicy actor.dist(obs) and critic.forward(global_state), original
Tobias station masks, role sharing conventions and ValueNorm.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def deep_encoder(input_dim: int, width: int = 128, output_dim: int = 64) -> nn.Sequential:
    """Three hidden Linear layers with widths 128,128,64 by default."""
    if input_dim < 1 or width < 8 or output_dim < 8:
        raise ValueError('Invalid MLP embedding dimensions')
    return nn.Sequential(
        nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width),
        nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width),
        nn.Linear(width, output_dim), nn.GELU(),
    )


def _pool(inputs: torch.Tensor, valid: torch.Tensor):
    """Masked mean/max; invalid fully padded tokens never contribute."""
    if inputs.ndim != 3 or valid.shape != inputs.shape[:2]:
        raise ValueError('Invalid pooling shapes')
    mask = valid.unsqueeze(-1)
    mean = (inputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    smallest = torch.finfo(inputs.dtype).min / 4
    maximum = inputs.masked_fill(~mask, smallest).max(dim=1).values
    maximum = torch.where(valid.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return mean, maximum


def _check_inputs(agvs: torch.Tensor, stations: torch.Tensor, feature_dim: int):
    if (agvs.ndim != 3 or stations.ndim != 3 or
            agvs.shape[0] != stations.shape[0] or
            agvs.shape[-1] != feature_dim or stations.shape[-1] != feature_dim):
        raise ValueError('Expected [B,A,F] AGV and [B,S,F] station arrays with matching feature dimension')
    if agvs.shape[1] < 1 or stations.shape[1] < 1:
        raise ValueError('Requires at least one AGV and station slot')
    if not torch.isfinite(agvs).all() or not torch.isfinite(stations).all():
        raise ValueError('Non-finite observation')


class CandidateAwareDispatchActor(nn.Module):
    """Dedicated deep per-AGV/per-station encoders and per-station MLP logits.

    EGO is row 0 after Tobias shuffling. Station ID is a fixed action-slot ID.
    Action mask is the unchanged original 'operation_first' mask; it has
    precedence over all logits. No global state/critic information is used.
    """
    def __init__(self, feature_dim: int, n_stations: int = 27,
                 width: int = 128, embedding: int = 64, station_id_dim: int = 16):
        super().__init__()
        if feature_dim < 1 or n_stations < 2 or station_id_dim < 1:
            raise ValueError('Invalid candidate-aware dispatch dimensions')
        self.feature_dim = int(feature_dim)
        self.action_dim = int(n_stations)
        self.embedding = int(embedding)
        self.ego_encoder = deep_encoder(feature_dim, width, embedding)
        self.other_encoder = deep_encoder(2*feature_dim, width, embedding)
        self.station_identity = nn.Embedding(n_stations, station_id_dim)
        nn.init.normal_(self.station_identity.weight, std=0.02)
        # Station encoder knows current deciding AGV and this station identity.
        self.station_encoder = deep_encoder(2*feature_dim + station_id_dim,
                                             width, embedding)
        # [ego, other mean/max, station mean/max, station_j, ego*station_j]
        self.scorer = nn.Sequential(
            nn.Linear(7*embedding, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, embedding), nn.GELU(),
            nn.Linear(embedding, 1),
        )

    def logits(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        agvs, stat, action_mask = obs['agvs'], obs['stat'], obs['action_mask']
        _check_inputs(agvs, stat, self.feature_dim)
        b, a, _ = agvs.shape
        n = self.action_dim
        if stat.shape[1] != n or action_mask.shape != (b, n):
            raise ValueError('Mismatch in station count vs action mask')
        if not torch.isfinite(action_mask).all() or not (action_mask > .5).any(dim=1).all():
            raise ValueError('Missing or nonfinite valid Dispatch action')
        ego = agvs[:, 0]
        e = self.ego_encoder(ego)
        other = agvs[:, 1:]
        if a > 1:
            o = self.other_encoder(torch.cat((ego[:, None].expand(-1,a-1,-1), other), dim=-1))
            om, ox = _pool(o, other.abs().sum(-1) > 0)
        else:
            om = ox = torch.zeros_like(e)
        station_ids = torch.arange(n,device=stat.device)
        station_ident = self.station_identity(station_ids)[None].expand(b,-1,-1)
        joint = torch.cat((ego[:,None].expand(-1,n,-1), stat, station_ident),dim=-1)
        se = self.station_encoder(joint)
        # A legal station is present even if its raw feature row is all zero.
        station_valid = (stat.abs().sum(-1) > 0) | (action_mask > .5)
        sm, sx = _pool(se, station_valid)
        global_features = torch.cat((e, om, ox, sm, sx), dim=-1)
        global_features = global_features[:, None].expand(-1,n,-1)
        pair = torch.cat((global_features, se, e[:,None]*se), dim=-1)
        logits = self.scorer(pair).squeeze(-1)
        if logits.shape != (b,n) or not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite or incorrectly shaped Dispatch logits')
        return logits

    def dist(self, obs: dict[str, torch.Tensor]) -> Categorical:
        from thesis.mappo.networks import masked_logits
        return Categorical(logits=masked_logits(self.logits(obs), obs['action_mask']))


class DeepCentralizedCritic(nn.Module):
    """A three-hidden-layer (128,128,64) global value head per MAPPO role.

    Input is the *unshuffled* physical state and physical ego index from
    Matrix.get_mappo_state(). No Actor observation shuffle or future info.
    """
    def __init__(self, feature_dim: int, width: int = 128, embedding: int = 64):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.agv_encoder = deep_encoder(feature_dim, width, embedding)
        self.station_encoder = deep_encoder(feature_dim, width, embedding)
        self.value_head = nn.Sequential(
            nn.Linear(5*embedding, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, embedding), nn.GELU(),
            nn.Linear(embedding, 1),
        )

    def forward(self, global_state: dict[str, torch.Tensor]) -> torch.Tensor:
        agvs, stat, ego_idx = (global_state['agvs'], global_state['stat'], global_state['ego'])
        _check_inputs(agvs, stat, self.feature_dim)
        b, a, _ = agvs.shape
        if (ego_idx.shape != (b,) or ego_idx.dtype != torch.long or
                (ego_idx < 0).any() or (ego_idx >= a).any()):
            raise ValueError('Critic needs physical ego indices LongTensor[B] within fleet')
        agv_embed = self.agv_encoder(agvs)
        station_embed = self.station_encoder(stat)
        rows = torch.arange(b, device=agvs.device)
        ego_embedding = agv_embed[rows, ego_idx]
        other_valid = (agvs.abs().sum(-1) > 0).clone()
        other_valid[rows, ego_idx] = False
        other_mean, other_max = _pool(agv_embed, other_valid)
        stat_mean, stat_max = _pool(station_embed, stat.abs().sum(-1) > 0)
        value = self.value_head(torch.cat((ego_embedding, other_mean, other_max,
                                          stat_mean, stat_max), dim=-1)).squeeze(-1)
        if value.shape != (b,) or not torch.isfinite(value).all():
            raise RuntimeError('Non-finite value network output')
        return value


def install_v14(model, feature_dim: int, n_stations: int,
                width: int = 128, embedding: int = 64, station_id_dim: int = 16):
    """Install all 2 Dispatch Actors + all 3 Critics with FRESH optimizer/ValueNorm.

    Routing Actor and its optimizer remain the fresh V12 GAT4 network.
    NEVER call after loading pretrained checkpoints. Safe only before EP0.
    """
    from thesis.mappo.algorithm import RunningValueNorm
    required = {'routing', 'dispatch_front', 'dispatch_rear'}
    if set(model.roles) != required:
        raise ValueError('Expected routing, dispatch_front, dispatch_rear role keys')
    out = {}
    for name, role in model.roles.items():
        if name != 'routing':
            if int(role.actor.action_dim) != n_stations:
                raise ValueError('Unexpected station count for Dispatch role')
            role.actor = CandidateAwareDispatchActor(feature_dim, n_stations,
                                width, embedding, station_id_dim).to(model.device)
            role.actor_optimizer = torch.optim.Adam(role.actor.parameters(), lr=model.cfg.actor_lr)
        role.critic = DeepCentralizedCritic(feature_dim, width, embedding).to(model.device)
        role.critic_optimizer = torch.optim.Adam(role.critic.parameters(),lr=model.cfg.critic_lr)
        role.valuenorm = RunningValueNorm()
        out[name] = {'actor': 'v12_congestion_edge_gat4' if name == 'routing' else 'candidate_aware_deep_mlp',
                     'critic': 'global_deep_mlp_128_128_64'}
    for name, role in model.roles.items():
        if any(not p.requires_grad for net in (role.actor,role.critic) for p in net.parameters()):
            raise RuntimeError('Frozen actor/critic weights found')
        if name != 'routing':
            if any(x.startswith(('base.','encoder.main','station_attention')) for x in role.actor.state_dict()):
                raise RuntimeError('Old Transformer/pooled Dispatch encoder not fully replaced')
    return out
