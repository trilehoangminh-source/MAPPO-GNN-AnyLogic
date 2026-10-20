"""Read-only rollout GNN mapping metrics and PPO consistency preflight.

Based on research-practice lessons from EPyMARL (reproducible evaluation)
and TA-RWARE (explicit agent/target geometry diagnostics). It does NOT change
policy outputs, rewards, action masking, environment state or AnyLogic logic.
"""
from __future__ import annotations

from collections import Counter
from types import MethodType

import numpy as np
import torch


class RoutingGraphAudit:
    def __init__(self, actor, sample_limit: int = 2048):
        from thesis.mappo.gnn_routing_B3_v121 import GraphRoutingActor
        if not isinstance(actor, GraphRoutingActor):
            raise TypeError('RoutingGraphAudit requires B3 graph routing actor')
        if hasattr(actor, '_uoe_audit_installed'):
            raise RuntimeError('Routing audit already installed')
        self.actor = actor
        self.sample_limit = max(0, int(sample_limit))
        self.reset()
        old_logits = actor.logits

        def monitored_logits(this_actor, observation):
            # Only collect when acting/evaluating without grad, not during PPO updates.
            if not torch.is_grad_enabled():
                self.observe(observation)
            return old_logits(observation)

        actor.logits = MethodType(monitored_logits, actor)
        actor._uoe_audit_installed = True

    def reset(self):
        self.count = Counter()
        self.nearest_ego = []
        self.nearest_target = []
        self.nearest_candidate = []

    def observe(self, obs):
        with torch.no_grad():
            agvs = obs['agvs']
            mask = obs['action_mask']
            if agvs.ndim != 3 or mask.ndim != 2 or mask.shape[-1] != 5:
                raise ValueError('Invalid Routing observation dimensions')
            if not torch.isfinite(agvs).all() or not torch.isfinite(mask).all():
                raise ValueError('Nonfinite GNN input/mask')
            if torch.any((mask > 0.5).sum(dim=-1) == 0):
                raise RuntimeError('Routing action mask has no legal action')
            ego = agvs[:, 0]
            now, target = ego[:, 4:6], ego[:, 6:8]
            neighbors = ego[:, 8:16].reshape(-1, 4, 2)
            pos_id, pos_ok = self.actor._match(now.unsqueeze(1))
            tar_id, tar_ok = self.actor._match(target.unsqueeze(1))
            mov_id, mov_ok = self.actor._match(neighbors)
            del pos_id, tar_id, mov_id
            legal = mask[:, 1:] > 0.5
            total = int(agvs.shape[0])
            self.count['routing_decisions'] += total
            self.count['ego_exact_node'] += int(pos_ok.sum())
            self.count['target_exact_node'] += int(tar_ok.sum())
            self.count['legal_move_slots'] += int(legal.sum())
            self.count['mapped_legal_move_slots'] += int((mov_ok & legal).sum())
            self.count['decisions_with_unmapped_legal_move'] += int(((~mov_ok) & legal).any(-1).sum())
            self.count['fallback_ego'] += total - int(pos_ok.sum())
            self.count['fallback_target'] += total - int(tar_ok.sum())
            nodes = self.actor.node_xy
            def record(coords, output, selected=None):
                if len(output) >= self.sample_limit:
                    return
                distance = torch.linalg.vector_norm(coords.unsqueeze(-2) - nodes, dim=-1).min(-1).values
                if selected is not None:
                    distance = distance[selected]
                take = min(self.sample_limit - len(output), distance.numel())
                output.extend(distance.detach().reshape(-1)[:take].cpu().tolist())
            record(now, self.nearest_ego)
            record(target, self.nearest_target)
            record(neighbors, self.nearest_candidate, legal)

    def report(self):
        c = dict(self.count)
        def rate(num, denom):
            return round(c.get(num, 0) / max(c.get(denom, 0), 1), 6)
        def qs(items):
            if not items:
                return None
            return {'p50': round(float(np.percentile(items, 50)), 6),
                    'p95': round(float(np.percentile(items, 95)), 6)}
        c['ego_map_rate'] = rate('ego_exact_node', 'routing_decisions')
        c['target_map_rate'] = rate('target_exact_node', 'routing_decisions')
        c['candidate_map_rate'] = rate('mapped_legal_move_slots', 'legal_move_slots')
        c['nearest_node_distance'] = {
            'ego': qs(self.nearest_ego), 'target': qs(self.nearest_target),
            'legal_candidate': qs(self.nearest_candidate)}
        c['note'] = ('Exact-node match uses original B3-GNN 1e-3 coordinate tolerance; '
                     'fallback statistics are diagnostics, not proof of failed learning.')
        return c


def verify_ppo_rollout_parity(model, trajectories, *, max_samples: int = 64,
                              atol: float = 1e-4):
    """Compare stored behavior logp with SAME current policy before first PPO step.

    This catches distribution inconsistencies caused by missing Dijkstra bias,
    action mask changes or BatchNorm train/eval mismatch. Do not call after update.
    """
    import thesis.mappo.algorithm as algo
    role = model.roles['routing']
    was_training = role.actor.training
    role.actor.eval()
    samples = [s for traj in trajectories for s in traj if s.get('role') == 'routing']
    selected = samples[:max(0, int(max_samples))]
    errors = []
    mask_bad = 0
    try:
        with torch.no_grad():
            for step in selected:
                obs = step['observation']
                if 'routing_guide_bias' not in obs:
                    raise RuntimeError('Routing sample missing Dijkstra guidance in PPO buffer')
                action = int(step['action'])
                mask = np.asarray(obs['action_mask']).reshape(-1)
                if action < 0 or action >= len(mask) or mask[action] <= 0.5:
                    mask_bad += 1
                    continue
                tensor_obs = algo.batchify(algo.as_actor(obs, model.device))
                lp = float(role.actor.dist(tensor_obs).log_prob(
                    torch.tensor([action], device=model.device)).item())
                errors.append(abs(lp - float(step['logp'])))
    finally:
        role.actor.train(was_training)
    worst = max(errors, default=0.0)
    result = dict(routing_samples=len(selected), compared=len(errors),
                  mask_violations=mask_bad, max_logprob_error=worst,
                  tolerance=atol, passed=mask_bad == 0 and worst <= atol)
    if not result['passed']:
        raise RuntimeError('STOP before PPO update: rollout/current logprob mismatch: '+str(result))
    return result


def verify_dispatch_backbone(model):
    from thesis.mappo.gnn_routing_B3_v121 import GraphRoutingActor
    dispatch = [k for k in model.roles if k.startswith('dispatch')]
    if not dispatch:
        raise RuntimeError('No dispatch policy found')
    if any(isinstance(model.roles[k].actor, GraphRoutingActor) for k in dispatch):
        raise RuntimeError('Dispatching must remain NON-GNN')
    return dispatch
