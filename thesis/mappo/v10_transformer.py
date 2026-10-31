"""V10 residual attention adapters for Tobias/AnyLogic MAPPO.

No original V8 code is modified. Each V10 adapter wraps a pretrained V8
actor/critic and starts with EXACTLY the same output (zero-initialized head).
Attention tokens: valid AGVs plus station rows; invalid/padded rows are masked.

The Transformer is a *spatial/set attention* encoder, not a time-series model.
Both input and action formats are identical to V8; the original PPO trainer,
mask and role-based centralized training can be reused.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def _entity_mask(agvs: torch.Tensor, stations: torch.Tensor,
                 action_mask: torch.Tensor | None = None) -> torch.Tensor:
    """True for real keys. Always keep ego, and allowed station tokens.

    A permitted station may have all-zero features in a legacy observation:
    do NOT hide it simply because its feature vector is zero.
    """
    valid_agv = agvs.abs().sum(dim=-1) > 0
    valid_agv = valid_agv.clone()
    valid_agv[:, 0] = True
    valid_stat = stations.abs().sum(dim=-1) > 0
    if action_mask is not None:
        if stations.shape[1] != action_mask.shape[1]:
            raise ValueError('Station tokens must match Dispatch action dimension')
        valid_stat = valid_stat | (action_mask > 0.5)
    return torch.cat([valid_agv, valid_stat], dim=1)


def _encoder(dim: int, heads: int, layers: int) -> nn.TransformerEncoder:
    if dim < 4 or heads < 1 or dim % heads or layers < 1:
        raise ValueError('Invalid transformer dimensions/heads/layers')
    block = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=dim * 2,
        dropout=0.0, activation='gelu', batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)


class ResidualStationTransformerActor(nn.Module):
    """V8 MLP logits + Transformer station-specific correction.

    Wrapping existing pretrained actor guarantees identical initial policy.
    This keeps the action space and original operation-first mask unchanged.
    """
    def __init__(self, pretrained_actor: nn.Module, feature_dim: int,
                 n_stations: int, hidden: int = 32, heads: int = 2, layers: int = 1):
        super().__init__()
        self.base = pretrained_actor
        self.action_dim = int(n_stations)
        self.agv_embedding = nn.Linear(feature_dim, hidden)
        self.station_embedding = nn.Linear(feature_dim, hidden)
        self.agent_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.station_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.encoder = _encoder(hidden, heads, layers)
        self.station_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # Strict policy parity with V8 at the beginning of the experiment.
        nn.init.zeros_(self.station_head[-1].weight)
        nn.init.zeros_(self.station_head[-1].bias)

    def logits(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        agvs, stat, mask = obs['agvs'], obs['stat'], obs['action_mask']
        if agvs.ndim != 3 or stat.ndim != 3 or mask.ndim != 2:
            raise ValueError('Expected batched AGVs [B,A,F], stations [B,S,F], mask [B,S]')
        if stat.shape[1] != self.action_dim or mask.shape[1] != self.action_dim:
            raise ValueError('Station count and action mask do not match')
        if agvs.shape[0] != stat.shape[0]:
            raise ValueError('Different AGV/Station batch sizes')
        # The V8 Actor forward path uses this exact encoder + head.
        base_logits = self.base.policy(self.base.encoder(obs))
        if base_logits.shape != mask.shape:
            raise ValueError('Legacy Dispatch head dimension changed')
        tokens = torch.cat([
            self.agv_embedding(agvs) + self.agent_type,
            self.station_embedding(stat) + self.station_type,
        ], dim=1)
        valid = _entity_mask(agvs, stat, mask)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        ego = encoded[:, :1, :].expand(-1, self.action_dim, -1)
        station = encoded[:, agvs.shape[1]:, :]
        pair = torch.cat([ego, station, ego * station], dim=-1)
        correction = self.station_head(pair).squeeze(-1)
        return base_logits + correction

    def dist(self, obs: dict[str, torch.Tensor]) -> Categorical:
        from thesis.mappo.networks import masked_logits
        return Categorical(logits=masked_logits(self.logits(obs), obs['action_mask']))


class ResidualTransformerCentralCritic(nn.Module):
    """V8 centralized pooled critic plus ego-aware global Transformer residual.

    Keeps V8 ValueNorm and V8 initial values intact. By default installed
    on dispatch roles only so Routing and Residual Edge-GAT remain frozen.
    """
    def __init__(self, pretrained_critic: nn.Module, feature_dim: int,
                 hidden: int = 32, heads: int = 2, layers: int = 1):
        super().__init__()
        self.base = pretrained_critic
        self.agv_embedding = nn.Linear(feature_dim, hidden)
        self.station_embedding = nn.Linear(feature_dim, hidden)
        self.agent_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.station_type = nn.Parameter(torch.zeros(1, 1, hidden))
        self.encoder = _encoder(hidden, heads, layers)
        self.value_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        agvs, stat, ego_index = state['agvs'], state['stat'], state['ego']
        base_val = self.base(state)
        if ego_index.ndim != 1 or ego_index.shape[0] != agvs.shape[0]:
            raise ValueError('Invalid ego indices')
        if ((ego_index < 0) | (ego_index >= agvs.shape[1])).any():
            raise ValueError('Centralized critic ego index out of range')
        tokens = torch.cat([
            self.agv_embedding(agvs) + self.agent_type,
            self.station_embedding(stat) + self.station_type,
        ], dim=1)
        valid = _entity_mask(agvs, stat)
        valid = valid.clone()
        batch_idx = torch.arange(agvs.shape[0], device=agvs.device)
        valid[batch_idx, ego_index] = True
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        ego_repr = encoded[batch_idx, ego_index]
        # Global context uses masked mean to ignore padded AGVs/stations.
        global_repr = (encoded * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        correction = self.value_head(torch.cat([ego_repr, global_repr], dim=-1)).squeeze(-1)
        return base_val + correction


def install_v10(model, variant: str, feature_dim: int, n_stations: int,
                hidden: int = 32, heads: int = 2, layers: int = 1) -> None:
    """Install *after loading* the V8 source checkpoint, before V10 checkpoint.

    Variants: control=V8 MLP, A=Transformer Dispatch actor, B=Transformer
    Dispatch critic, C=both. Routing network is NEVER replaced.
    """
    if variant not in ('control', 'A', 'B', 'C'):
        raise ValueError('Variant must be control, A, B, or C')
    # Reinitialize Dispatch optimizers for EVERY branch, including control.
    # Otherwise the V8 MLP control would inherit Adam moments while V10
    # adapters start with new Adam states, confounding architecture ablations.
    for name in ('dispatch_front', 'dispatch_rear'):
        role = model.roles[name]
        if variant in ('A', 'C'):
            role.actor = ResidualStationTransformerActor(
                role.actor, feature_dim, n_stations, hidden, heads, layers).to(model.device)
        if variant in ('B', 'C'):
            role.critic = ResidualTransformerCentralCritic(
                role.critic, feature_dim, hidden, heads, layers).to(model.device)
        role.actor_optimizer = torch.optim.Adam(role.actor.parameters(), lr=model.cfg.actor_lr)
        role.critic_optimizer = torch.optim.Adam(role.critic.parameters(), lr=model.cfg.critic_lr)
