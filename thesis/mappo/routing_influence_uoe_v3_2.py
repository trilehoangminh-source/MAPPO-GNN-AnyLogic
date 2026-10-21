"""Read-only B3-UOE-V3.2 audit: GAT raw logits vs Dijkstra-guided logits.

Run during evaluation only, AFTER model.act(role='routing') has selected an action.
This module does not change the chosen action, gradients, mask, rewards or policy.
It also inspects the actor-visible positions of other AGVs to diagnose the
V3.1 proxy that counted exactly seven neighbors at every decision.
"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import json

import numpy as np
import torch


class RoutingInfluenceAudit:
    def __init__(self, *, radius=6.0, strict_logp_atol=1e-4):
        self.radius = float(radius)
        self.strict_logp_atol = float(strict_logp_atol)
        if not 0 < self.radius < 1e6 or not 0 < self.strict_logp_atol < 1:
            raise ValueError('Invalid radius or logp tolerance')
        self.counters = Counter()
        self._attached = False
        self._out = None

    @torch.no_grad()
    def _row(self, model, obs, state, action, logp, greedy):
        from thesis.mappo import algorithm as algo
        actor = model.roles['routing'].actor
        if actor.training:
            raise RuntimeError('Influence audit must only run in evaluation mode (actor.eval())')
        ten = algo.batchify(algo.as_actor(obs, model.device))
        mask = ten['action_mask'] > 0.5
        bias = ten.get('routing_guide_bias')
        if bias is None:
            raise RuntimeError('No routing_guide_bias: cannot compare raw/GAT logits')
        raw = actor.logits(ten)
        if raw.shape != (1, 5) or bias.shape != raw.shape or mask.shape != raw.shape:
            raise RuntimeError('Raw actor logits, mask, and prior must be [1,5]')
        # The original actor's dist uses masked_logits; normalization does not
        # affect softmax, so computing it this way is equivalent on valid actions.
        raw_masked = raw.masked_fill(~mask, -1e9)
        guided_masked = (raw_masked + bias.masked_fill(~mask, 0.0))
        probs_raw = raw_masked.softmax(-1)[0]
        probs_guided = guided_masked.softmax(-1)[0]
        action = int(action)
        if not bool(mask[0, action]):
            raise RuntimeError('Selected action violates the input action mask')
        new_logp = float(probs_guided[action].log().item())
        error = abs(new_logp - float(logp))
        if error > self.strict_logp_atol:
            raise RuntimeError(f'Guided logprob parity failed: {error:.6g}')
        raw_choice = int(probs_raw.argmax().item())
        guided_choice = int(probs_guided.argmax().item())
        if greedy and action != guided_choice:
            raise RuntimeError('Evaluation action differs from reconstructed guided argmax')
        gids = ((bias[0] > 0) & mask[0]).nonzero().flatten().tolist()
        agvs = np.asarray(obs['agvs'], np.float64)
        if agvs.ndim != 2 or agvs.shape[1] < 16:
            raise RuntimeError('Expected actor-visible Tobias AGV observation')
        ego_xy = agvs[0, 4:6]
        other = agvs[1:]
        live = np.any(np.abs(other)>0, axis=1)
        other_xy = other[live, 4:6]
        other_dist = np.linalg.norm(other_xy-ego_xy[None, :], axis=-1) if len(other_xy) else np.empty(0)
        # Do not call these physical vehicle-to-vehicle distances: the observation
        # stores LAST-NODE coordinates, which may differ from continuous positions.
        dmin = float(other_dist.min()) if len(other_dist) else None
        dmax = float(other_dist.max()) if len(other_dist) else None
        row = dict(schema='B3_UOE_V3_2_GAT_INFLUENCE',
                   sim_time_hint=None,
                   physical_agv=int(state['ego']),
                   action=action, greedy=bool(greedy),
                   valid_actions=mask[0].nonzero().flatten().tolist(),
                   guide_actions=gids, guide_bias=[float(x) for x in bias[0].cpu().tolist()],
                   raw_logits=[float(x) for x in raw[0].cpu().tolist()],
                   raw_argmax=raw_choice, guided_argmax=guided_choice,
                   action_equals_raw=(action==raw_choice),
                   action_equals_guided=(action==guided_choice),
                   raw_argmax_is_guide=raw_choice in gids if gids else None,
                   guided_argmax_is_guide=guided_choice in gids if gids else None,
                   raw_guide_probability=float(probs_raw[gids].sum().item()) if gids else None,
                   guided_guide_probability=float(probs_guided[gids].sum().item()) if gids else None,
                   selected_raw_probability=float(probs_raw[action].item()),
                   selected_guided_probability=float(probs_guided[action].item()),
                   logprob_parity_error=error,
                   others_count=int(len(other_xy)),
                   other_last_node_distance_min=dmin,
                   other_last_node_distance_max=dmax,
                   nearby_last_nodes_1m=int(np.count_nonzero(other_dist<=1.0)),
                   nearby_last_nodes_2m=int(np.count_nonzero(other_dist<=2.0)),
                   nearby_last_nodes_6m=int(np.count_nonzero(other_dist<=6.0)),
                   nearby_last_nodes_radius=int(np.count_nonzero(other_dist<=self.radius)),
                   identical_last_nodes=int(np.count_nonzero(other_dist<1e-6)),
                   first_three_other_xy=other_xy[:3].tolist())
        self.counters['decisions'] += 1
        self.counters['guided_available'] += int(bool(gids))
        self.counters['raw_top1_is_guide'] += int(bool(gids) and raw_choice in gids)
        self.counters['guided_top1_is_guide'] += int(bool(gids) and guided_choice in gids)
        self.counters['guide_changes_top1'] += int(raw_choice != guided_choice)
        self.counters['has_alternative'] += int(bool(gids) and any(i not in gids for i in row['valid_actions']))
        self.counters['all_others_within_radius'] += int(len(other_dist)>0 and np.all(other_dist<=self.radius))
        self.counters['some_identical_last_node'] += int(row['identical_last_nodes'] > 0)
        self.counters['stop_available'] += int(0 in row['valid_actions'])
        return row

    @contextmanager
    def attach(self, model, output_file):
        if self._attached:
            raise RuntimeError('Routing influence audit already attached')
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.counters.clear()
        original = model.act
        self._attached = True
        try:
            with path.open('x', encoding='utf-8') as handle:
                def audited(_model, role, obs, state, greedy=False):
                    result = original(role, obs, state, greedy=greedy)
                    if role == 'routing':
                        action, logp, _value = result
                        row = self._row(_model, obs, state, action, logp, greedy)
                        handle.write(json.dumps(row, allow_nan=False) + '\n')
                    return result
                model.act = MethodType(audited, model)
                try:
                    yield self
                finally:
                    model.act = original
        finally:
            self._attached = False

    def report(self):
        result = dict(self.counters)
        result['notes'] = [
            'Counterfactual raw vs guided action is for the same OBSERVATION, not the same future trajectory.',
            'Derived logits audited in evaluation mode only; no gradients or optimizer calls.',
            'Nearby distances use actor-visible last-node coordinates; not physical collision distance.',
            'Guidance is a soft additive logit prior; do not infer policy improvement from this audit.',
        ]
        return result
