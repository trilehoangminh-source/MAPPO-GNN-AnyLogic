"""Read-only routing decision instrumentation for Tobias AnyLogic / B3-UOE-V2.

The route trace is OBSERVATIONAL: it does not change actions, action masks,
policy logits, reward, the environment, gradients or PPO training.

Inspired by research evaluation practice in EPyMARL and separation of task
assignment from motion planning in TA-RWARE; independent implementation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, isfinite
from pathlib import Path
from types import MethodType
import json

import numpy as np

STAT_KEYS = ("block", "liveLock", "wrongTarget", "targetReached", "jobDone")


def _as_float_list(values):
    return [float(x) for x in values]


class DirectedRouteDistances:
    """Shortest remaining distance on *directed* Hive arcs, by real node ID.

    GraphSpec's ``arcs`` contain consecutive indices in ``spec.ids``. Avoid
    accidentally confusing those indices with AnyLogic node IDs.
    """
    def __init__(self, spec, tolerance: float = 1e-3):
        self.ids = tuple(int(x) for x in spec.ids)
        self.xy = np.asarray(spec.xy, dtype=np.float64)
        self.arcs = {(int(i), int(j)) for i, j in spec.arcs}
        self.edges = set()
        self.reverse = defaultdict(list)
        self.tolerance = float(tolerance)
        if not len(self.ids) or self.xy.shape != (len(self.ids), 2):
            raise ValueError("Invalid graph spec")
        if not np.isfinite(self.xy).all() or not 0 < self.tolerance < 1:
            raise ValueError("Invalid node coordinates / tolerance")
        for i, j in self.arcs:
            if not (0 <= i < len(self.ids) and 0 <= j < len(self.ids)):
                raise ValueError("Arc index out of range")
            w = hypot(*(self.xy[i] - self.xy[j]))
            if w <= 0:
                raise ValueError("Zero-length arc not supported")
            self.edges.add((self.ids[i], self.ids[j]))
            self.reverse[self.ids[j]].append((self.ids[i], w))
        self._cache = {}

    def match(self, xy):
        try:
            xy = np.asarray(xy, dtype=np.float64).reshape(2)
        except (ValueError, TypeError):
            return None
        if not np.isfinite(xy).all():
            return None
        dist = np.abs(self.xy - xy).max(-1)
        i = int(np.argmin(dist))
        return self.ids[i] if float(dist[i]) < self.tolerance else None

    def distance_map(self, target):
        if target is None:
            return {}
        if target not in self._cache:
            d = {target: 0.0}
            heap = [(0.0, target)]
            while heap:
                cost, v = heappop(heap)
                if cost > d[v] + 1e-12:
                    continue
                for u, weight in self.reverse.get(v, ()):
                    newcost = cost + weight
                    if newcost < d.get(u, float('inf')) - 1e-12:
                        d[u] = newcost
                        heappush(heap, (newcost, u))
            self._cache[target] = d
        return self._cache[target]


class RoutingDecisionTrace:
    """Records one row for each *observed* Routing decision in evaluation.

    The trace installs a temporary wrapper over ``model.act``. The wrapper
    calls the original method exactly ONCE, then records data from the same
    actor-visible observation. ``model.act`` is restored even on exceptions.
    """
    def __init__(self, graph_spec, *, local_radius: float = 6.0):
        if not 0 < local_radius < 100000:
            raise ValueError("local_radius must be positive")
        self.graph = DirectedRouteDistances(graph_spec)
        self.radius = float(local_radius)
        self.count = Counter()
        self._original = None
        self._stream = None
        self._last_stats = None
        self._last_time = None
        self._by_agent_target = {}

    def reset(self):
        self.count.clear()
        self._last_stats = None
        self._last_time = None
        self._by_agent_target.clear()

    def _snapshot(self, env):
        stats = getattr(env, 'statistics', None)
        if not isinstance(stats, dict):
            try:
                stats = dict(stats or {})
            except (TypeError, ValueError):
                stats = {}
        result = {}
        for key in STAT_KEYS:
            value = stats.get(key)
            try:
                v = float(value)
            except (ValueError, TypeError):
                v = None
            result[key] = v if v is not None and isfinite(v) else None
        return result

    def _record(self, env, obs, state, action, logp, value):
        agvs = np.asarray(obs['agvs'], dtype=np.float64)
        mask = np.asarray(obs['action_mask'], dtype=np.float64).reshape(-1)
        guide = np.asarray(obs.get('routing_guide_bias'), dtype=np.float64).reshape(-1)
        if agvs.ndim != 2 or agvs.shape[0] < 1 or agvs.shape[1] < 16:
            raise ValueError('Routing observation invalid: expected [AGVs, >=16]')
        if mask.shape != (5,) or guide.shape != (5,):
            raise ValueError('Routing mask and guidance must each have five actions')
        if not (np.isfinite(agvs).all() and np.isfinite(mask).all() and np.isfinite(guide).all()):
            raise ValueError('Non-finite routing trace data')
        if not (0 <= int(action) < 5 and mask[int(action)] > .5):
            raise RuntimeError('Action selected outside original actor action mask')

        ego = agvs[0]
        current_xy, target_xy = ego[4:6], ego[6:8]
        candidates_xy = ego[8:16].reshape(4, 2)
        current = self.graph.match(current_xy)
        target = self.graph.match(target_xy)
        candidates = [self.graph.match(xy) if mask[i + 1] > .5 else None
                      for i, xy in enumerate(candidates_xy)]
        remaining = self.graph.distance_map(target)
        dist_before = remaining.get(current)
        chosen_next = candidates[action-1] if action > 0 else None
        dist_after = remaining.get(chosen_next)
        # The shortest path to the target is directed; no Euclidean shortcut.
        distance_change = (round(float(dist_after - dist_before), 5)
                           if (dist_before is not None and dist_after is not None and action > 0)
                           else None)
        guide_slots = [int(i) for i in range(5) if guide[i] > 0 and mask[i] > .5]
        chosen_is_guide = (int(action) in guide_slots) if guide_slots else None
        legal_slots = [int(i) for i in range(5) if mask[i] > .5]
        # This proximity is a LAST-NODE proxy, not collision detection or true
        # continuous AGV separation. Exclude completely padded entities.
        other = agvs[1:]
        other_existing = np.any(np.abs(other) > 0, axis=1)
        other_positions = other[other_existing, 4:6]
        other_dists = (np.linalg.norm(other_positions - current_xy[None, :], axis=1)
                       if len(other_positions) else np.empty(0))
        nearby = int(np.sum(other_dists <= self.radius))
        near_candidate = []
        for xy in candidates_xy:
            near_candidate.append(int(np.sum(np.linalg.norm(
                other_positions - xy[None, :], axis=1) < self.radius))
                                if len(other_positions) else 0)
        physical = int(state['ego'])
        sim_time = float(env.get_mappo_time())
        if self._last_time is not None and sim_time + 1e-6 < self._last_time:
            raise RuntimeError('Simulation clock unexpectedly moved backwards')
        self._last_time = sim_time
        stats = self._snapshot(env)
        # Deltas are measured between routing observations, not per-action causal
        # effects! AnyLogic may have multiple events or dispatches in between.
        if self._last_stats is None:
            deltas = {k: None for k in STAT_KEYS}
        else:
            deltas = {k: (max(0., v - self._last_stats[k])
                           if v is not None and self._last_stats[k] is not None else None)
                      for k, v in stats.items()}
        self._last_stats = stats
        last_target = self._by_agent_target.get(physical)
        target_changed = last_target is not None and target != last_target
        self._by_agent_target[physical] = target
        row = dict(schema='B3_UOE_V3_1_route_trace',
                   time_s=round(sim_time, 5), physical_agv=physical,
                   current_node=current, target_node=target, next_node=chosen_next,
                   current_xy=_as_float_list(current_xy), target_xy=_as_float_list(target_xy),
                   candidate_nodes=candidates, action=int(action),
                   valid_actions=legal_slots, guide_actions=guide_slots,
                   selected_is_guided=chosen_is_guide,
                   action_edge_valid=(current, chosen_next) in self.graph.edges if action > 0 else None,
                   directed_remaining_before=round(float(dist_before), 5) if dist_before is not None else None,
                   directed_remaining_after=round(float(dist_after), 5) if dist_after is not None else None,
                   remaining_change=distance_change,
                   other_agvs_last_node_nearby=nearby,
                   other_agvs_last_node_near_candidates=near_candidate,
                   target_changed_for_physical_slot=target_changed,
                   action_logp=float(logp), estimated_value=float(value),
                   cumulative_stats=stats, stats_increase_since_previous_routing_observation=deltas)
        self._stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.count['routing_decisions'] += 1
        self.count['stops'] += int(action == 0)
        self.count['with_guide'] += int(bool(guide_slots))
        self.count['selected_guide'] += int(chosen_is_guide is True)
        self.count['unmapped_current'] += int(current is None)
        self.count['unmapped_target'] += int(target is None)
        self.count['unmapped_selected_move'] += int(action > 0 and chosen_next is None)
        self.count['missing_directed_arc_for_selected_move'] += int(action > 0 and
             current is not None and chosen_next is not None and (current, chosen_next) not in self.graph.edges)
        self.count['reachable_selected_moves'] += int(action > 0 and dist_after is not None)
        self.count['moving_toward_target'] += int(distance_change is not None and distance_change < -1e-6)
        self.count['moving_away_from_target'] += int(distance_change is not None and distance_change > 1e-6)
        self.count['target_changes_observed'] += int(target_changed)
        self.count['decisions_near_other_agvs'] += int(nearby > 0)

    @contextmanager
    def attach(self, model, env, file_path):
        if self._original is not None:
            raise RuntimeError('Tracer is already attached')
        self.reset()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._original = model.act
        original = self._original
        try:
            with path.open('x', encoding='utf-8') as stream:
                self._stream = stream
                def wrapped(model_instance, role, obs, state, greedy=False):
                    # Exactly one original call. All decision logic is unchanged.
                    action, logp, value = original(role, obs, state, greedy=greedy)
                    if role == 'routing':
                        self._record(env, obs, state, action, logp, value)
                    return action, logp, value
                model.act = MethodType(wrapped, model)
                try:
                    yield self
                finally:
                    model.act = original
                    self._stream = None
        finally:
            self._original = None

    def report(self):
        c = dict(self.count)
        denominator = max(1, self.count.get('with_guide', 0))
        c['guide_follow_rate'] = round(self.count.get('selected_guide', 0)/denominator, 6)
        c['local_radius_m'] = self.radius
        c['notes'] = [
            'Action is the actual sampled/greedy Routing action, not a proposed action.',
            'Dijkstra is a soft prior; disagreement is not automatically an error.',
            'Proximity uses other AGVs last-node actor-visible coordinates, not true collision geometry.',
            'Cumulative block/livelock increments since the preceding Routing observation are associations, not causal labels.',
            'Target changes for a physical slot may be valid new tasks or PettingZoo alias resets.',
            'No neural network, mask, policy, reward or simulator behaviour is modified.',
        ]
        return c
