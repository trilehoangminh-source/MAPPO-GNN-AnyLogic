#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""PASSIVE V15 EP100 Dispatch diagnostic (seed 910006 vs 910007).

Place next to scripts/evaluate_v15_ep100_buffer.py in the V15 project.
The existing evaluation script is reused for its exact EP100 checkpoint, topology
and fingerprint verification and for the original counters. This module NEVER
changes a chosen action, mask, policy weight, reward, AnyLogic model or config.

From the project root (PowerShell):
  .\.venv311\Scripts\python.exe .\scripts\diagnose_v15_ep100_dispatch.py `
    --stage-dir '.\mappo_runs\V15_ROUTING5_NOMASK_MB256_FIXED\joint16' --dry-run

  .\.venv311\Scripts\python.exe .\scripts\diagnose_v15_ep100_dispatch.py `
    --stage-dir '.\mappo_runs\V15_ROUTING5_NOMASK_MB256_FIXED\joint16' `
    --port 51885

Outputs in <stage-dir>/dispatch_diag_ep0100_910006_910007/:
 - dispatch_decisions.csv (one row per dispatch)
 - candidate_logits.csv (one row per available station or buffer per decision,
   including masked-out choices for audit; logit/probability/mask)
 - repeated_targets.csv (consecutive dispatches per physical AGV)
 - station_preference_summary.csv (per-station selection and eligibility)
 - comparison_by_seed.csv (JobDone, Block, LiveLock and diagnostic aggregates)
 - notes.json (methodology, baseline comparison, source/checkpoint hash)

Limitations: the AnyLogic station state vector is logged RAW; its bit-to-state
mapping is NOT assumed. 'Station occupied' is a nonmoving-AGV-at-node PROXY.
An unchanged next-node at two dispatch events does NOT prove an AGV did not move
in between. Decisions are event-sampled, not a continuous trajectory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from evaluate_v15_ep100_buffer import (  # locally verified existing V15 evaluator
    TrafficProbe, read_json, sha256, same_xy, numeric, write_csv,
)

SEEDS = (910006, 910007)
# Geometry stations represent DIFFERENT production steps; never treat them as peers.
TYPE_BY_PREFIX = ('wps_', 'hsn1_', 'hsn2_', 'fls_')


def station_group(name: str) -> str:
    name = name.lower()
    for p in TYPE_BY_PREFIX:
        if name.startswith(p):
            return p.rstrip('_')
    if name.startswith('puffer'):
        return 'buffer'
    return name


def softmax_masked(raw_logits, mask):
    """Reconstruct the DISPATCH categorical probabilities only; do not sample."""
    raw = np.asarray(raw_logits, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=np.float32).reshape(-1) > 0.5
    if raw.shape != valid.shape or not valid.any():
        raise ValueError(f'Bad dispatch mask/logits shapes: {raw.shape} vs {valid.shape}')
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite dispatch logits')
    output = np.zeros_like(raw)
    z = raw[valid] - np.max(raw[valid])
    exp = np.exp(z)
    output[valid] = exp / exp.sum()
    return output


def xy_d(x, y):
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size != 2 or b.size != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return float(np.linalg.norm(a - b))


def slim_float(v, digits=7):
    return round(float(v), digits)


class DispatchDiagnostic(TrafficProbe):
    """Adds read-only post-action introspection to the original TrafficProbe."""

    def __init__(self, adapter, env, seed):
        super().__init__(adapter, env, seed)
        self.action_rows = []
        self.decision_rows = []
        self.repeat_rows = []
        self.last_dispatch = {}  # Physical AGV -> previous dispatch observation
        self.serial = 0
        self.eligibility = Counter()
        self.chosen = Counter()
        self.mask_present_count = Counter()
        self.legal_unpicked = Counter()
        self.busy_proxy = Counter()
        self.same_target_proxy = Counter()
        self.less_loaded_peer_chosen = 0
        self.less_loaded_peer_events = 0
        self.same_choice_repeats = 0
        self.same_choice_no_sampled_node_progress = 0

    def _catalog_with_names(self):
        if self.station_catalog is None:
            self._build_catalog()
        if not hasattr(self, 'node_names'):
            from thesis.envs.part_variants import MatrixPart
            part = MatrixPart()
            names = {}
            for name, nodes in part.stat_node.items():
                if isinstance(nodes, (tuple, list)):
                    for k, node in enumerate(nodes):
                        names[int(node)] = f'{name}_{k+1}'
                else:
                    names[int(nodes)] = name
            self.node_names = {int(node): names.get(int(node), f'node_{node}')
                               for node in self.station_catalog}

    def _get_logits(self, role, obs):
        """Forward in eval/no_grad AFTER actual decision; never touch policy RNG.

        This network uses Linear/BatchNorm/ReLU only (no dropout); eval forward
        yields the same logits as the decision forward, without optimizer steps.
        """
        import torch
        from thesis.mappo.algorithm import as_actor, batchify
        net = self.inner.roles[role]
        if net.actor.training:
            raise RuntimeError('Dispatch actor unexpectedly in train mode after act()')
        with torch.no_grad():
            obs_t = batchify(as_actor(obs, net.device))
            logits = net.actor.logits(obs_t).detach().cpu().numpy().reshape(-1)
        return logits

    def act(self, role, obs, state, greedy=False):
        # This is the SINGLE ORIGINAL action-selection call. Every calculation
        # below runs AFTER that call; none of it can alter the action returned.
        action, logp, value = super().act(role, obs, state, greedy=greedy)
        if role not in ('dispatch_front', 'dispatch_rear'):
            return action, logp, value
        self._catalog_with_names()
        self.serial += 1
        if not greedy:
            raise RuntimeError('Diagnostic must run in evaluation mode (greedy=True)')
        index = int(action)
        node = int(self.station_catalog[index])
        now = numeric(self.env.get_mappo_time())
        ego = int(state['ego'])
        agv = np.asarray(state['agvs'], dtype=float)
        stations = np.asarray(state['stat'], dtype=float)
        mask = np.asarray(obs['action_mask'], dtype=float).reshape(-1)
        raw = self._get_logits(role, obs)
        if len(raw) != len(self.station_catalog) or len(mask) != len(raw):
            raise RuntimeError(f'Action mismatch: catalog={len(self.station_catalog)}, '
                               f'logits={len(raw)}, mask={len(mask)}')
        probs = softmax_masked(raw, mask)
        if mask[index] <= 0.5:
            raise RuntimeError('Chosen Dispatch action is masked out')
        if index != int(np.argmax(probs)):
            raise RuntimeError(f'Greedy Dispatch mismatch: action={index}, argmax={np.argmax(probs)}')

        target_counts, occupied = self._analyze_agvs(state)
        xy = self.station_xy[node]
        now_node_xy = np.asarray(agv[ego, 4:6], dtype=float)
        last_node_xy = np.asarray(agv[ego, 2:4], dtype=float)
        previous_dest_xy = np.asarray(agv[ego, 6:8], dtype=float)
        moving = int(agv[ego, 1] > .5)
        num_real_agvs = int((agv[:, 0] > 0.5).sum())
        ego_previous_target = any(same_xy(previous_dest_xy, self.station_xy[n])
                                  for n in self.station_catalog)
        state_source = str(getattr(self.env, '_mappo_station_source', 'unknown'))

        # A fresh/stale/unknown station observation is still logged RAW. Do not
        # claim which index is 'working' without confirming AnyLogic enum order.
        other_targeted, other_occupied = {}, {}
        candidate_masked, station_pos = {}, {}
        for j, dest in enumerate(self.station_catalog):
            station_pos[dest] = j
            this_target = int(target_counts[dest])
            if same_xy(previous_dest_xy, self.station_xy[dest]):
                this_target = max(0, this_target - 1)  # exclude ego
            other_targeted[dest] = this_target
            other_occupied[dest] = int(any(k != ego for k in occupied[dest]))
            candidate_masked[dest] = bool(mask[j] > .5)

        # Within identical machine types, compare only currently legal peers.
        chosen_name = self.node_names[node]
        chosen_group = station_group(chosen_name)
        peers = [n for n in self.station_catalog if n != node
                 and station_group(self.node_names[n]) == chosen_group
                 and candidate_masked[n]]
        less_loaded = [n for n in peers if other_targeted[n] < other_targeted[node]]
        unoccupied_peers = [n for n in peers if not other_occupied[n]]
        less_loaded_and_unoccupied = [n for n in less_loaded if not other_occupied[n]]
        if less_loaded:
            self.less_loaded_peer_events += 1
        if less_loaded_and_unoccupied:
            self.less_loaded_peer_chosen += 1

        prior = self.last_dispatch.get(ego)
        same_choice = prior is not None and prior['node'] == node
        no_sampled_node_change = prior is not None and same_xy(prior['next_node'], now_node_xy)
        if same_choice:
            self.same_choice_repeats += 1
            if no_sampled_node_change:
                self.same_choice_no_sampled_node_progress += 1
        prior_dist = (xy_d(prior['next_node'], xy) if prior is not None else None)
        curr_dist = xy_d(now_node_xy, xy)
        improving = None if prior_dist is None else int(curr_dist + 1e-6 < prior_dist)
        repeat = {
            'seed': self.seed, 'decision_id': self.serial, 'time_s': slim_float(now, 4),
            'physical_agv': ego, 'current_destination': chosen_name,
            'previous_destination': prior['name'] if prior else '',
            'same_destination_as_previous_dispatch': int(bool(same_choice)),
            'elapsed_since_previous_dispatch_s': '' if prior is None else slim_float(now - prior['time'], 4),
            'previous_sampled_next_node': '' if prior is None else json.dumps(prior['next_node']),
            'current_sampled_next_node': json.dumps(now_node_xy.tolist()),
            'same_sampled_next_node': '' if prior is None else int(no_sampled_node_change),
            'previous_distance_to_current_destination_norm': '' if prior_dist is None else slim_float(prior_dist),
            'current_distance_to_current_destination_norm': '' if curr_dist is None else slim_float(curr_dist),
            'sampled_distance_improved': '' if improving is None else improving,
            'at_destination_node_proxy': int(same_xy(now_node_xy, xy) and not moving),
            'warning': 'Sampled positions only; no continuous trajectory',
        }
        self.repeat_rows.append(repeat)
        self.last_dispatch[ego] = {'node': node, 'name': chosen_name,
                                   'time': now, 'next_node': now_node_xy.tolist()}

        for j, dest in enumerate(self.station_catalog):
            name = self.node_names[dest]
            valid = int(mask[j] > .5)
            self.mask_present_count[name] += 1
            self.eligibility[name] += valid
            self.chosen[name] += int(j == index)
            if valid and j != index:
                self.legal_unpicked[name] += 1
            self.busy_proxy[name] += int(valid and other_occupied[dest] > 0)
            self.same_target_proxy[name] += int(valid and other_targeted[dest] > 0)
            # Registered station rows have coordinates then statechart bits.
            # The station's *original* raw state vector is retained; expanded
            # with padding; no erroneous hardcoded 'busy' or 'working' field.
            st = np.asarray(stations[j], dtype=float).tolist() if j < len(stations) else []
            active_bits = [k for k, v in enumerate(st[2:], start=2)
                           if float(v) > 0.5] if st else []
            self.action_rows.append({
                'seed': self.seed, 'decision_id': self.serial,
                'time_s': slim_float(now, 4), 'physical_agv': ego,
                'role': role, 'action_index': j, 'node': dest, 'station': name,
                'group': station_group(name), 'is_buffer': int(station_group(name) == 'buffer'),
                'is_chosen': int(j == index), 'is_legal': valid,
                'raw_logit': slim_float(raw[j]),
                'masked_probability': slim_float(probs[j], 10),
                'other_agvs_targeting_candidate': other_targeted[dest],
                'other_stopped_agvs_at_candidate_proxy': len([k for k in occupied[dest] if k != ego]),
                'station_obs_source': state_source,
                'station_state_active_indices_raw': json.dumps(active_bits),
                'station_obs_vector_raw': json.dumps([round(float(v), 6) for v in st]),
                'station_busy_verified': '',  # intentionally absent
            })

        legal_ranking = sorted((j for j in range(len(mask)) if mask[j] > .5),
                               key=lambda j: probs[j], reverse=True)
        top3 = [self.node_names[self.station_catalog[j]] for j in legal_ranking[:3]]
        row = {
            'seed': self.seed, 'decision_id': self.serial,
            'time_s': slim_float(now, 4), 'physical_agv': ego,
            'role': role, 'chosen_station': chosen_name, 'chosen_node': node,
            'chosen_action_index': index, 'chosen_raw_logit': slim_float(raw[index]),
            'chosen_probability': slim_float(probs[index], 10),
            'greedy_top3_stations': '|'.join(top3),
            'legal_station_count': int((mask > .5).sum()),
            'legal_buffer_count': sum(mask[j] > .5 for j, nd in enumerate(self.station_catalog)
                                      if station_group(self.node_names[nd]) == 'buffer'),
            'mask_indices_legal': json.dumps([int(j) for j, x in enumerate(mask) if x > .5]),
            'chosen_group': chosen_group,
            'chosen_other_agvs_targeting': other_targeted[node],
            'chosen_other_stopped_at_node_proxy': int(other_occupied[node]),
            'legal_same_group_peers': '|'.join(self.node_names[n] for n in peers),
            'legal_same_group_less_targeted': '|'.join(self.node_names[n] for n in less_loaded),
            'legal_same_group_unoccupied_proxy': '|'.join(self.node_names[n] for n in unoccupied_peers),
            'legal_same_group_less_targeted_and_unoccupied': '|'.join(self.node_names[n] for n in less_loaded_and_unoccupied),
            'station_obs_source': state_source,
            'jobdone_global_at_decision': numeric((self.env.statistics or {}).get('jobDone')),
            'ego_moving': moving, 'ego_in_system': int(agv[ego, 0] > .5),
            'ego_last_node_normalized_xy': json.dumps(last_node_xy.tolist()),
            'ego_next_node_normalized_xy': json.dumps(now_node_xy.tolist()),
            'ego_target_before_new_dispatch_normalized_xy': json.dumps(previous_dest_xy.tolist()),
            'ego_prev_target_looked_like_station': int(ego_previous_target),
            'sampled_distance_to_chosen_station_norm': '' if curr_dist is None else slim_float(curr_dist),
            'same_destination_as_previous_dispatch': int(bool(same_choice)),
            'same_sampled_next_node_as_previous_dispatch': '' if prior is None else int(no_sampled_node_change),
            'number_active_agvs': num_real_agvs,
        }
        self.decision_rows.append(row)
        return action, logp, value

    def aggregate(self, env_summary):
        stats = env_summary.get('statistics', {})
        decision_count = len(self.decision_rows)
        return {
            'seed': self.seed, 'jobDone': numeric(stats.get('jobDone')),
            'block': numeric(stats.get('block')),
            'liveLock': numeric(stats.get('liveLock')),
            'targetReached': numeric(stats.get('targetReached')),
            'end_reason': str(env_summary.get('end_reason')),
            'dispatch_decisions': decision_count,
            'buffer_dispatches': sum(1 for r in self.decision_rows if r['chosen_group'] == 'buffer'),
            'chosen_legal_peer_less_targeted_event_count': self.less_loaded_peer_events,
            'chosen_legal_peer_less_targeted_unoccupied_event_count': self.less_loaded_peer_chosen,
            'same_destination_repeated_count': self.same_choice_repeats,
            'same_destination_and_unchanged_sampled_next_node_count': self.same_choice_no_sampled_node_progress,
            'observations_fresh_dispatch_count': sum(1 for r in self.decision_rows if r['station_obs_source'] == 'fresh_dispatch'),
        }


def _read_prior_summary(stage, ckhash, seed):
    p = stage / 'buffer_eval_ep0100' / 'summary_by_seed.csv'
    if not p.exists():
        return None
    with p.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if int(row['seed']) == seed:
                if row.get('checkpoint_sha256') != ckhash:
                    raise RuntimeError('EP100 checkpoint differs from earlier buffer evaluation: ' + str(p))
                return row
    return None


def _assert_parity(ours, previous):
    if previous is None:
        return {'prior_found': False, 'agrees': None}
    discrepancies = {}
    compare = [('jobDone', 'jobdone'), ('block', 'block'),
               ('liveLock', 'liveLock'), ('targetReached', 'targetReached'),
               ('dispatch_decisions', 'dispatch_decisions')]
    for ours_name, prior_name in compare:
        a = float(ours[ours_name])
        b = float(previous[prior_name])
        if not math.isclose(a, b, abs_tol=1e-5, rel_tol=0):
            discrepancies[ours_name] = (a, b)
    if discrepancies:
        raise RuntimeError(f'Evaluation changed unexpectedly: {discrepancies}. '
                           'Do not analyze logs until RNG / source parity is resolved.')
    return {'prior_found': True, 'agrees': True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-dir', type=Path, required=True)
    p.add_argument('--episode', type=int, default=100)
    p.add_argument('--port', type=int, default=51885)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-compare-existing', action='store_true',
                   help='Skip parity verification against buffer_eval_ep0100/summary_by_seed.csv')
    args = p.parse_args(argv)
    if args.episode != 100:
        p.error('This diagnostic is restricted to EP100 to prevent checkpoint confusion')
    if not (1 <= args.port <= 65535):
        p.error('Port must be 1..65535')

    stage = args.stage_dir.resolve()
    meta, state = read_json(stage / 'experiment.json'), read_json(stage / 'state.json')
    if meta.get('phase') != 'joint16':
        raise RuntimeError('Only joint16 is supported; got ' + str(meta.get('phase')))
    ck = stage / 'snapshot_ep_0100.pt'
    if not ck.is_file():
        raise FileNotFoundError(f'Missing EP100 snapshot: {ck}')
    if int(state.get('last_episode', 0)) < 100:
        raise RuntimeError('EP100 has not been completed')
    ckhash = sha256(ck)
    registered = {str(Path(s).resolve()) for s in state.get('snapshots', [])}
    if registered and str(ck.resolve()) not in registered:
        raise RuntimeError('EP100 snapshot not registered in stage/state.json')
    bp, sim = Path(meta['blueprint_file']).resolve(), Path(meta['model_file']).resolve()
    if sha256(bp) != meta['blueprint_sha256']:
        raise RuntimeError('Blueprint changed since training')
    if sha256(sim) != meta['anylogic_sha256']:
        raise RuntimeError('AnyLogic model changed since training')
    checks = {
        'architecture': ROOT / 'thesis/mappo/v15_tobias_arch.py',
        'environment_builder': ROOT / 'scripts/v15_tobias_build.py',
        'staged_policy': ROOT / 'scripts/v15_staged_collector.py',
        'ppo_algorithm': ROOT / 'thesis/mappo/algorithm.py',
        'ppo_stability': ROOT / 'thesis/mappo/mappo_stability_v7.py',
        'event_collector': ROOT / 'thesis/mappo/collector_stop_ab.py',
        'matrix_environment': ROOT / 'thesis/envs/matrix_v121.py',
        'behaviors': ROOT / 'thesis/envs/behaviors_v121.py',
        'alpyne_bridge': ROOT / 'thesis/envs/alpyne121_bridge.py',
    }
    for key, file in checks.items():
        expected = meta.get('fingerprints', {}).get(key)
        if expected and sha256(file) != expected:
            raise RuntimeError(f'Changed source {key}: {file}')
    output = (args.output or stage / 'dispatch_diag_ep0100_910006_910007').resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite existing diagnostic: ' + str(output))
    if not args.no_compare_existing:
        for seed in SEEDS:
            _read_prior_summary(stage, ckhash, seed)

    print('[EP100 DISPATCH DIAGNOSTIC]', 'seeds', SEEDS, 'checkpoint sha256', ckhash,
          'device', meta['device'], flush=True)
    if args.dry_run:
        print('[DRY RUN OK] Verified EP100, files, unchanged source and simulation hashes.')
        return

    import torch
    from v15_tobias_build import build_v15, TOBIAS_DISPATCH_MASK_MODE
    from v15_staged_collector import StagePolicyAdapter, PHASES
    from thesis.mappo.collector_stop_ab import EventCollector
    if str(meta['device']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; evaluation device must match training')
    os.environ['MAPPO_DISPATCH_MASK_MODE'] = TOBIAS_DISPATCH_MASK_MODE
    os.environ['MAPPO_MAX_MINIBATCH'] = str(meta['max_minibatch'])
    torch.set_num_threads(int(meta['cpu_threads']))
    random.seed(meta['base_seed']); np.random.seed(meta['base_seed']); torch.manual_seed(meta['base_seed'])
    env = None
    records, all_actions, all_decisions, all_repeats = [], [], [], []
    aggregate_counts = defaultdict(Counter)
    all_probability_stats = defaultdict(list)
    parity_rows = []
    try:
        env, model, graph, arch = build_v15(
            read_json(bp), model_path=sim, fleet_size=PHASES['joint16'][0],
            port=args.port, device=meta['device'], seed=meta['base_seed'],
            max_events=meta['max_events'], sim_seconds=3600,
            checkpoint=ck, phase='joint16')
        if not bool(env.config.obs_include_agv_target):
            raise RuntimeError('AGV target coordinates absent; cannot compute congestion proxies')
        adapter = StagePolicyAdapter(model, 'joint16', meta['base_seed'])
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            adapter.rng = np.random.default_rng(seed)
            probe = DispatchDiagnostic(adapter, env, seed)
            collector = EventCollector(probe, env, stop_mask_mode='baseline')
            with torch.no_grad():
                _, summary = collector.episode(seed=seed, train=False,
                                               max_event_steps=meta['max_events'])
            if summary.get('end_reason') == 'max_event_steps':
                raise RuntimeError(f'Seed {seed} truncated by max_events; refusing partial results')
            record = probe.aggregate(summary)
            previous = None if args.no_compare_existing else _read_prior_summary(stage, ckhash, seed)
            parity = _assert_parity(record, previous)
            records.append(record)
            parity_rows.append({'seed': seed, **parity})
            all_actions.extend(probe.action_rows)
            all_decisions.extend(probe.decision_rows)
            for arow in probe.action_rows:
                if arow['is_legal']:
                    all_probability_stats[(seed, arow['station'])].append(float(arow['masked_probability']))
            all_repeats.extend(probe.repeat_rows)
            for n in probe.mask_present_count:
                aggregate_counts[(seed, n)]['events_seen'] = probe.mask_present_count[n]
                aggregate_counts[(seed, n)]['legal_events'] = probe.eligibility[n]
                aggregate_counts[(seed, n)]['chosen'] = probe.chosen[n]
                aggregate_counts[(seed, n)]['legal_but_unselected'] = probe.legal_unpicked[n]
                aggregate_counts[(seed, n)]['legal_when_occupied_proxy'] = probe.busy_proxy[n]
                aggregate_counts[(seed, n)]['legal_when_other_agvs_targeted'] = probe.same_target_proxy[n]
            print(f"[EP100 seed={seed}] JobDone={record['jobDone']:.0f} "
                  f"Block={record['block']:.0f} LiveLock={record['liveLock']:.0f} "
                  f"Dispatch={record['dispatch_decisions']} "
                  f"repeat_same_dest={record['same_destination_repeated_count']} "
                  f"alt_less_congested={record['chosen_legal_peer_less_targeted_event_count']} "
                  f"parity={'MATCH' if parity['agrees'] else 'NOT_CHECKED'}", flush=True)
            for important in ('wps_1', 'wps_2', 'wps_3', 'hsn1_1', 'hsn1_2'):
                print(f"  {important:8s} legal={probe.eligibility[important]:4d} "
                      f"chosen={probe.chosen[important]:4d}", flush=True)
    finally:
        if env is not None:
            env.close()

    # Only write after BOTH seeds complete and all available parity checks pass.
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'dispatch_decisions.csv', all_decisions)
    write_csv(output/'candidate_logits.csv', all_actions)
    write_csv(output/'repeated_targets.csv', all_repeats)
    write_csv(output/'comparison_by_seed.csv', records)
    write_csv(output/'station_preference_summary.csv', [
        {'seed': seed, 'station': n, **values,
         'mean_prob_when_legal': round(float(np.mean(all_probability_stats[(seed, n)])), 6)
               if all_probability_stats[(seed, n)] else '',
         'max_prob_when_legal': round(float(max(all_probability_stats[(seed, n)])), 6)
               if all_probability_stats[(seed, n)] else ''}
        for (seed, n), values in sorted(aggregate_counts.items())])
    (output/'notes.json').write_text(json.dumps({
        'checkpoint_sha256': ckhash, 'episode': 100, 'seeds': SEEDS,
        'policy': 'unchanged greedy joint16 EP100',
        'actor_updates': 0, 'anylogic_changes': 0, 'reward_changes': 0,
        'mask_mode': str(TOBIAS_DISPATCH_MASK_MODE),
        'baseline_parity': parity_rows,
        'definitions': {
            'is_legal': 'Original Dispatch action mask at decision time',
            'raw_logit': 'Unmasked Dispatch Actor network output, BEFORE softmax',
            'masked_probability': 'Softmax restricted to legal Dispatch actions',
            'other_stopped_agvs_at_candidate_proxy': 'Stopped AGVs whose next-node equals station; NOT verified station working state',
            'station_obs_vector_raw': 'Original AnyLogic station observation. [0:2] normalized XY; following indices are unordered raw state flags, not assumed labelled',
            'station_obs_source': 'fresh_dispatch vs cached (from Matrix capture source)',
            'legal_same_group_less_targeted': 'Same-process station options with fewer AGVs targeting than chosen station; only among legally masked candidates',
            'same_sampled_next_node': 'Node reported at consecutive dispatch decisions; lack of change does NOT prove no movement between samples',
        },
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[COMPLETE] Passive diagnostic saved to', output, flush=True)
    print('Upload comparison_by_seed.csv, station_preference_summary.csv, '
          'dispatch_decisions.csv and repeated_targets.csv for analysis.', flush=True)


if __name__ == '__main__':
    main()
