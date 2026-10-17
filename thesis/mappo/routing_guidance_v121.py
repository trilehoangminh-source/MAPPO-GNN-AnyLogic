"""B3 experiment: MAPPO with a *soft*, directed-graph shortest-path prior.

Inspired by Peng (2023), MAPPOHR (arXiv:2306.01270). This module is our
Tobias/AnyLogic adaptation, NOT a reproduction of the paper's full planner.

Critical PPO invariant: exactly the same additive logit prior is stored in
rollout observations and used in PPO's recomputed action distributions.
No intervention is made AFTER selecting an action. The environment action mask
remains authoritative. No privileged critic-only data is fed to the actor.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import heapq
from math import hypot, isfinite
from types import MethodType
import numpy as np
import torch
from torch.distributions import Categorical
from .algorithm import MAPPO
from .collector_stop_ab import EventCollector


def _key(coord):
    return tuple(round(float(c), 6) for c in coord)


class DirectedGraphGuide:
    """Suggest closest-path legal next move, using Tobias's directed Hive graph.

    Each 1..4 move maps to the agent-visible neighbor coordinates at agvs[0,8:16].
    Target is agvs[0,6:8]. Edge directions are honored; no directed-edge-free
    Euclidean distance shortcut is used. Dynamic AGVs/blocks remain to MAPPO.
    """
    def __init__(self, hive, strength=1.5):
        self.hive = hive
        self.strength = float(strength)
        if not 0 <= self.strength <= 20:
            raise ValueError('routing guide strength must be between 0 and 20')
        self._cache_token = None
        self._coords_to_id = {}
        self._reverse = {}
        self._dist_cache = {}
        self.stats = Counter()

    def reset_stats(self):
        self.stats.clear()

    def _refresh(self):
        nodes, paths = self.hive.nodes, self.hive.paths
        if not nodes or not paths:
            return False
        token = (id(nodes), id(paths))
        if token == self._cache_token:
            return True
        self._cache_token = token
        self._coords_to_id = {_key(coords): int(i) for i, coords in nodes.items()}
        self._reverse = defaultdict(list)
        self._dist_cache.clear()
        for u, v in paths:  # directed path list; bidirectional arcs already expanded by Hive
            if int(u) not in nodes or int(v) not in nodes:
                raise ValueError(f'Bad graph arc: {(u,v)}')
            c1, c2 = nodes[u], nodes[v]
            w = hypot(float(c1[0])-float(c2[0]), float(c1[1])-float(c2[1]))
            if w > 0:
                self._reverse[int(v)].append((int(u), w))
        return True

    def _dist_to_target(self, target_id):
        if target_id not in self._dist_cache:
            distance = {target_id: 0.0}
            todo = [(0.0, target_id)]
            while todo:
                d, node = heapq.heappop(todo)
                if d > distance[node] + 1e-12:
                    continue
                for previous, weight in self._reverse.get(node, ()):
                    cand = d + weight
                    if cand + 1e-12 < distance.get(previous, float('inf')):
                        distance[previous] = cand
                        heapq.heappush(todo, (cand, previous))
            self._dist_cache[target_id] = distance
        return self._dist_cache[target_id]

    def suggest(self, observation):
        """Return 5-element logit prior and diagnostic tag. Never mask moves."""
        self.stats['routing_decisions'] += 1
        prior = np.zeros(5, dtype=np.float32)
        if not self._refresh():
            self.stats['missing_graph'] += 1
            return prior, 'missing_graph'
        agvs = np.asarray(observation['agvs'], dtype=np.float32)
        mask = np.asarray(observation['action_mask'], dtype=np.float32).reshape(-1)[:5]
        if agvs.ndim != 2 or agvs.shape[1] < 16:
            raise ValueError('Routing observation lacks current/target/neighbor coordinates')
        ego = agvs[0]
        current_id = self._coords_to_id.get(_key(ego[4:6]))
        target_id = self._coords_to_id.get(_key(ego[6:8]))
        if current_id is None or target_id is None:
            self.stats['unmapped_coordinates'] += 1
            return prior, 'unmapped_coordinates'
        if current_id == target_id:
            self.stats['at_target'] += 1
            return prior, 'at_target'
        # The four movement action slots are in Tobias's original observation order.
        distances = self._dist_to_target(target_id)
        valid_candidates = []
        for idx in range(1, 5):
            if mask[idx] <= 0.5:
                continue
            next_id = self._coords_to_id.get(_key(ego[8 + (idx-1)*2 : 10 + (idx-1)*2]))
            if next_id is None or next_id not in distances:
                continue
            # Only guide available outgoing arcs, never hallucinate a transition.
            if not any(prev == current_id for prev, _ in self._reverse.get(next_id, ())):
                continue
            cx,cy = self.hive.nodes[current_id]
            nx,ny = self.hive.nodes[next_id]
            local = hypot(float(nx)-float(cx), float(ny)-float(cy))
            valid_candidates.append((idx, distances[next_id] + local))
        if not valid_candidates:
            self.stats['no_legal_path'] += 1
            return prior, 'no_legal_path'
        minimum = min(distance for _,distance in valid_candidates)
        best = [idx for idx,d in valid_candidates if abs(d-minimum)<1e-6]
        for idx in best:
            prior[idx] = self.strength
        self.stats['guided'] += 1
        self.stats['choices_guided'] += len(best)
        return prior, 'guided'


def enable_routing_distribution_prior(model):
    """Patch ONLY this model's Routing distribution and MAPPO's tensor adapter.

    No changes to neural parameters/architecture, so checkpoints retain the
    normal MAPPO shape. The static guide prior must be supplied via each stored
    routing observation during BOTH sampling and updates/evaluation.
    """
    import thesis.mappo.algorithm as algo
    if not getattr(algo.as_actor, '_b3_guidance', False):
        original_adapter = algo.as_actor
        def _guided_adapter(obs, device):
            result = original_adapter(obs, device)
            if 'routing_guide_bias' in obs:
                result['routing_guide_bias'] = torch.as_tensor(
                    obs['routing_guide_bias'], dtype=torch.float32, device=device)
            return result
        _guided_adapter._b3_guidance = True
        algo.as_actor = _guided_adapter

    actor = model.roles['routing'].actor
    if getattr(actor, '_b3_guide_dist', False):
        return
    original_dist = actor.dist
    def _guided_dist(self, obs):
        if 'routing_guide_bias' not in obs:
            raise RuntimeError('B3 routing observation missing guidance; PPO would be inconsistent')
        dist = original_dist(obs)
        bias = obs['routing_guide_bias']
        if bias.shape != dist.logits.shape:
            raise ValueError(f'Guidance shape {bias.shape} != routing logits {dist.logits.shape}')
        if not torch.isfinite(bias).all():
            raise ValueError('Nonfinite shortest-path bias')
        # Original hard Action Mask still takes priority. Never unmask invalid.
        bias = bias.masked_fill(obs['action_mask'] <= 0.5, 0.0)
        return Categorical(logits=dist.logits + bias)
    actor.dist = MethodType(_guided_dist, actor)
    actor._b3_guide_dist = True


class GuidedMAPPO(MAPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        enable_routing_distribution_prior(self)
        self.guidance_stats = Counter()

    def act(self, role, obs, state, greedy=False):
        action, logp, value = super().act(role, obs, state, greedy)
        if role == 'routing':
            self.guidance_stats['routing_actions'] += 1
            prior = obs['routing_guide_bias']
            if np.any(prior > 0):
                self.guidance_stats['guided_actions'] += 1
                self.guidance_stats['selected_guide'] += int(prior[int(action)] > 0)
        return action, logp, value


class GuidedEventCollector(EventCollector):
    def __init__(self, model, environment, stop_mask_mode='forbid_when_move',
                 routing_guide_strength=1.5):
        super().__init__(model, environment, stop_mask_mode)
        self.guide = DirectedGraphGuide(environment.agent_hive, routing_guide_strength)

    def _prepare(self, observation, role):
        obs = super()._prepare(observation, role)
        if role == 'routing':
            bias, _ = self.guide.suggest(obs)
            obs['routing_guide_bias'] = bias.copy()
        return obs

    def episode(self, *args, **kwargs):
        self.guide.reset_stats()
        self.model.guidance_stats.clear()
        segments, meta = super().episode(*args, **kwargs)
        stats = dict(self.guide.stats)
        stats.update(dict(self.model.guidance_stats))
        meta['routing_guidance'] = stats
        if stats.get('routing_actions', 0) and stats.get('guided', 0) == 0:
            raise RuntimeError('B3 got no graph path guidance: check networkcontext and coordinate matching; do NOT train blindly')
        return segments, meta
