"""CTDE architectures compatible with original Matrix observations.

Actor input is the ORIGINAL transformed observation (agvs/stat/action_mask).
Critic input is a separate deterministic RAW physical state, provided by Matrix.
For the main experiment, select original_gnn and original_linear backbones.
A dependency-light pooled backbone is provided for CPU unit tests/ablations.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def masked_logits(logits, mask):
    """Safe hard action mask; never silently enable padding actions."""
    if mask.shape != logits.shape:
        raise ValueError(f"Action mask {tuple(mask.shape)} != logits {tuple(logits.shape)}")
    valid = mask > 0.5
    if (~valid.any(-1)).any():
        raise ValueError("Environment produced an action mask without valid actions")
    return logits.masked_fill(~valid, torch.finfo(logits.dtype).min / 2)


class EntityEncoder(nn.Module):
    """Permutation-invariant aggregation of other AGVs and stations.

    Separately embeds deciding AGV, other AGVs, and station features.
    Zero-padded entities are excluded from their mean/max aggregation.
    """
    def __init__(self, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.other = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.station = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.output_dim = hidden * 5

    def _pool(self, feats, mask):
        mask = mask.unsqueeze(-1)
        mean = (feats * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        neg = torch.finfo(feats.dtype).min / 4
        maximum = feats.masked_fill(~mask.bool(), neg).max(dim=1).values
        maximum = torch.where(mask.any(dim=1), maximum, torch.zeros_like(maximum))
        return mean, maximum

    def forward(self, obs):
        agvs, stat = obs['agvs'], obs['stat']
        ego = agvs[:, 0]
        others = agvs[:, 1:]
        ego_features = self.main(ego)
        if others.shape[1]:
            omean, omax = self._pool(self.other(others), others.abs().sum(-1) > 0)
        else:
            omean = omax = torch.zeros_like(ego_features)
        if stat.shape[1]:
            smean, smax = self._pool(self.station(stat), stat.abs().sum(-1) > 0)
        else:
            smean = smax = torch.zeros_like(ego_features)
        return torch.cat([ego_features, omean, omax, smean, smax], dim=-1)


class Actor(nn.Module):
    def __init__(self, obs_space, role: str, action_dim: int, backbone: str, hidden: int = 64, env_type: str = 'matrix'):
        super().__init__()
        self.role = role
        dim = obs_space['agvs'].shape[-1]
        if backbone == 'pooled':
            self.encoder = EntityEncoder(dim, hidden)
            encoding_dim = self.encoder.output_dim
        elif backbone == 'original_gnn' and role == 'routing':
            from thesis.policies.ma_gnn_routing import GNNFeatureExtractor
            self.encoder = GNNFeatureExtractor(
                env_type, obs_space, n_features=dim, embed_dim=hidden,
                with_stations=False, position_embedd_dim=0,
                ff_embedd_dim=4, n_convolutions=2,
            )
            encoding_dim = 5 * hidden
        elif backbone == 'original_linear' and role == 'dispatching':
            from thesis.policies.ma_lin_net import LinFE
            self.encoder = LinFE(
                obs_space, embed_dim=hidden, with_agvs=True,
                with_stations=True, position_embedd_dim=0,
                ff_embedd_dim=4,
            )
            encoding_dim = 5 * hidden
        else:
            raise ValueError(f'Unsupported actor backbone: {role=} {backbone=}')
        self.policy = nn.Sequential(nn.Linear(encoding_dim, hidden * 2), nn.Tanh(), nn.Linear(hidden * 2, action_dim))
        self.action_dim = action_dim

    def dist(self, observation):
        logits = self.policy(self.encoder(observation))
        return Categorical(logits=masked_logits(logits, observation['action_mask']))


class CentralCritic(nn.Module):
    """Value(s, ego) from unshuffled ALL-AGV/station physical state.

    Does not use actor's randomly shuffled/padded observation to fake CTDE.
    Critic sees the current physical AGV in absolute index order and every
    station. The role has its own critic/value normalizer.
    """
    def __init__(self, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.agv = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.stat = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(hidden * 5, hidden * 2), nn.Tanh(), nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    @staticmethod
    def pool(x, valid):
        m = valid.unsqueeze(-1)
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        neg = torch.finfo(x.dtype).min / 4
        maximum = x.masked_fill(~m, neg).max(1).values
        maximum = torch.where(m.any(1), maximum, torch.zeros_like(maximum))
        return mean, maximum

    def forward(self, global_state):
        agvs, stations, indices = global_state['agvs'], global_state['stat'], global_state['ego']
        if (indices < 0).any() or (indices >= agvs.shape[1]).any():
            raise ValueError('Global ego index outside AGV array')
        agv_emb = self.agv(agvs)
        stat_emb = self.stat(stations)
        ego = agv_emb[torch.arange(indices.shape[0], device=indices.device), indices]
        # Pool only OTHER AGVs; ego is encoded separately
        agv_valid = agvs.abs().sum(-1) > 0
        agv_valid = agv_valid.clone()
        agv_valid[torch.arange(len(indices), device=indices.device), indices] = False
        a_mean, a_max = self.pool(agv_emb, agv_valid)
        s_mean, s_max = self.pool(stat_emb, stations.abs().sum(-1) > 0)
        return self.head(torch.cat([ego, a_mean, a_max, s_mean, s_max], -1)).squeeze(-1)
