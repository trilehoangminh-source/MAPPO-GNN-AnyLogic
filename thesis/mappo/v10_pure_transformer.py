"""Independent Transformer dispatching networks for the Tobias AnyLogic MAPPO.

NO V8 Dispatch MLP module, output, parameter, or learned feature encoder is
used by either pure network. A frozen V8 checkpoint is used only to retain the
Routing Residual Edge-GAT policy (and optionally for the actor_only control's
critic). This is a set/spatial Transformer, not temporal or autoregressive.

Each station is scored by a shared attention-based scorer rather than by a
fixed-index Linear(H, n_stations) head. Station identity embeddings preserve
stable station IDs in the fixed AnyLogic topology; other AGV tokens receive
no position embedding, avoiding dependence on shuffled AGV order.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Categorical


def _check_shapes(agvs, stations):
    if agvs.ndim != 3 or stations.ndim != 3 or agvs.shape[0] != stations.shape[0]:
        raise ValueError('Expected AGVs [B,A,F], stations [B,S,F] with same batch size')
    if agvs.shape[-1] != stations.shape[-1]:
        raise ValueError('AGV/station feature dimensions differ')
    if not torch.isfinite(agvs).all() or not torch.isfinite(stations).all():
        raise ValueError('Non-finite actor/critic observations')


def _valid_entities(agvs, stations, ego_indices=None, action_mask=None):
    """True = token is present; always preserve decision maker and legal targets."""
    valid_agvs = (agvs.abs().sum(dim=-1) > 0).clone()
    if ego_indices is None:
        valid_agvs[:, 0] = True  # actor's ego always occupies shuffled row zero
    else:
        b = torch.arange(agvs.shape[0], device=agvs.device)
        valid_agvs[b, ego_indices] = True
    valid_stations = (stations.abs().sum(dim=-1) > 0)
    if action_mask is not None:
        if action_mask.shape != valid_stations.shape:
            raise ValueError('Dispatch mask must match station slots')
        valid_stations = valid_stations | (action_mask > 0.5)
    return torch.cat((valid_agvs, valid_stations), dim=1)


def _transformer(hidden, heads, layers):
    if hidden < 8 or heads < 1 or hidden % heads != 0 or layers < 1:
        raise ValueError('Require hidden >= 8, hidden divisible by heads, layers >= 1')
    # No dropout: PPO recomputes log-probs and requires rollout/update parity.
    block = nn.TransformerEncoderLayer(
        d_model=hidden, nhead=heads, dim_feedforward=hidden*2,
        dropout=0.0, activation='gelu', batch_first=True, norm_first=False)
    return nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)


class PureStationTransformerActor(nn.Module):
    """Fresh Transformer policy from observation -> n_stations masked logits.

    No pretrained_actor parameter is accepted or stored. Only self-attention,
    native Transformer FFNs and a shared station-scoring MLP are used.
    """
    def __init__(self, feature_dim: int, n_stations: int,
                 hidden: int=32, heads: int=2, layers: int=1):
        super().__init__()
        if feature_dim < 1 or n_stations < 2:
            raise ValueError('Invalid feature dimension/station count')
        self.action_dim = int(n_stations)
        self.agv_projection = nn.Linear(feature_dim, hidden)
        self.station_projection = nn.Linear(feature_dim, hidden)
        self.agv_type = nn.Parameter(torch.empty(1, 1, hidden).normal_(std=.02))
        self.station_type = nn.Parameter(torch.empty(1, 1, hidden).normal_(std=.02))
        self.ego_marker = nn.Parameter(torch.empty(1, 1, hidden).normal_(std=.02))
        self.station_identity = nn.Embedding(n_stations, hidden)
        nn.init.normal_(self.station_identity.weight, std=.02)
        self.embedding_norm = nn.LayerNorm(hidden)
        self.encoder = _transformer(hidden, heads, layers)
        # A separate score for EVERY station. NOT the old fixed-index V8 MLP.
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden*4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.hidden = hidden

    def logits(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        agvs, stations, mask = obs['agvs'], obs['stat'], obs['action_mask']
        _check_shapes(agvs, stations)
        batch, n_agvs, feat = agvs.shape
        if stations.shape[1] != self.action_dim or mask.shape != (batch, self.action_dim):
            raise ValueError('Action count/station rows/action mask mismatch')
        valid = _valid_entities(agvs, stations, action_mask=mask)
        if not (mask > 0.5).any(dim=1).all():
            raise ValueError('No valid Dispatch action in one or more observations')
        stations_ids = torch.arange(self.action_dim, device=stations.device)
        a = self.agv_projection(agvs) + self.agv_type
        a = a.clone()
        a[:, :1] = a[:, :1] + self.ego_marker
        s = (self.station_projection(stations) + self.station_type
             + self.station_identity(stations_ids)[None, :, :])
        tokens = self.embedding_norm(torch.cat((a, s), dim=1))
        h = self.encoder(tokens, src_key_padding_mask=~valid)
        ego = h[:, :1].expand(-1, self.action_dim, -1)
        station = h[:, n_agvs:]
        pair = torch.cat((ego, station, ego*station, ego-station), dim=-1)
        scores = self.pair_scorer(pair).squeeze(-1)
        if scores.shape != mask.shape:
            raise RuntimeError('Wrong station score shape')
        return scores

    def dist(self, obs: dict[str, torch.Tensor]) -> Categorical:
        from thesis.mappo.networks import masked_logits
        return Categorical(logits=masked_logits(self.logits(obs), obs['action_mask']))


class PureTransformerCentralCritic(nn.Module):
    """Independent global AGV/Station Transformer -> V(s,ego).

    Does not inherit from or hold a pooled V8 MLP critic. Its output is
    unnormalized; pre-existing RolePolicy ValueNorm performs normalizing.
    """
    def __init__(self, feature_dim: int, n_stations: int,
                 hidden: int=32, heads: int=2, layers: int=1):
        super().__init__()
        self.n_stations = int(n_stations)
        self.agv_projection = nn.Linear(feature_dim, hidden)
        self.station_projection = nn.Linear(feature_dim, hidden)
        self.agv_type = nn.Parameter(torch.empty(1, 1, hidden).normal_(std=.02))
        self.station_type = nn.Parameter(torch.empty(1, 1, hidden).normal_(std=.02))
        self.ego_marker = nn.Parameter(torch.empty(1, 1, hidden).normal_(std=.02))
        self.station_identity = nn.Embedding(n_stations, hidden)
        nn.init.normal_(self.station_identity.weight, std=.02)
        self.embedding_norm = nn.LayerNorm(hidden)
        self.encoder = _transformer(hidden, heads, layers)
        self.value_head = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        agvs, stations, indices = state['agvs'], state['stat'], state['ego']
        _check_shapes(agvs, stations)
        batch, n_agvs, _ = agvs.shape
        if indices.shape != (batch,) or indices.dtype != torch.long:
            raise ValueError('Critic ego index must be LongTensor[B]')
        if (indices < 0).any() or (indices >= n_agvs).any():
            raise ValueError('Critic ego index outside AGV array')
        if stations.shape[1] != self.n_stations:
            raise ValueError('Critic station count changed')
        valid = _valid_entities(agvs, stations, ego_indices=indices)
        row = torch.arange(batch, device=agvs.device)
        a = self.agv_projection(agvs) + self.agv_type
        markers = torch.zeros_like(a)
        markers[row, indices] = self.ego_marker[0, 0]
        a = a + markers
        ids = torch.arange(self.n_stations, device=stations.device)
        s = (self.station_projection(stations) + self.station_type
             + self.station_identity(ids)[None, :, :])
        tokens = self.embedding_norm(torch.cat((a,s),dim=1))
        h = self.encoder(tokens, src_key_padding_mask=~valid)
        ego = h[row, indices]
        context = (h*valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1,keepdim=True).clamp(min=1)
        return self.value_head(torch.cat((ego,context),dim=-1)).squeeze(-1)


def install_v10_pure(model, feature_dim: int, n_stations: int,
                     variant: str='both', hidden: int=32, heads: int=2, layers: int=1):
    """Replace Dispatch role nets AFTER restoring a frozen V8 checkpoint.

    both: reset BOTH Dispatch Actor and Centralized Critic, ValueNorm, Adam.
    actor_only: reset Dispatch Actor, keep V8 Dispatch Critic as ablation.
    Neither variant modifies Routing; default both has NO V8 Dispatch weights.
    """
    if variant not in ('both','actor_only'):
        raise ValueError('variant must be both or actor_only')
    from thesis.mappo.algorithm import RunningValueNorm
    for name in ('dispatch_front','dispatch_rear'):
        role = model.roles[name]
        role.actor = PureStationTransformerActor(
            feature_dim, n_stations, hidden, heads, layers).to(model.device)
        if variant == 'both':
            role.critic = PureTransformerCentralCritic(
                feature_dim, n_stations, hidden, heads, layers).to(model.device)
            role.valuenorm = RunningValueNorm()   # reset previous MLP critic normalization
        role.actor_optimizer = torch.optim.Adam(role.actor.parameters(), lr=model.cfg.actor_lr)
        role.critic_optimizer = torch.optim.Adam(role.critic.parameters(), lr=model.cfg.critic_lr)
        if any(k.startswith('base.') for k in role.actor.state_dict()):
            raise RuntimeError('V8 Dispatch MLP accidentally survived inside pure actor')
        if variant=='both' and any(k.startswith('base.') for k in role.critic.state_dict()):
            raise RuntimeError('V8 Dispatch MLP accidentally survived inside pure critic')
