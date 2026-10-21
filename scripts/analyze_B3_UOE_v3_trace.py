"""Offline aggregation of V3.1 Routing Decision Trace; NO AnyLogic dependency.

This analysis NEVER asserts that a routing action caused a blockage. AnyLogic
statistics are sampled at decision points and may reflect several previous
routing, dispatching and simulation events.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_rows(path):
    with Path(path).open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if row.get('schema') != 'B3_UOE_V3_1_route_trace':
                    raise ValueError(f'Unexpected trace schema {path}:{lineno}')
                yield row


def summarize(rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No routing decisions found')
    c = Counter()
    nodes = defaultdict(Counter)
    transitions = Counter()
    prev_target = {}
    for row in rows:
        c['routing_decisions'] += 1
        node = row.get('current_node')
        bucket = nodes[str(node) if node is not None else 'UNMAPPED']
        bucket['decisions'] += 1
        c['stops'] += int(row['action'] == 0)
        c['with_guide'] += int(bool(row['guide_actions']))
        c['guide_followed'] += int(row['selected_is_guided'] is True)
        bucket['guide_available'] += int(bool(row['guide_actions']))
        bucket['guide_followed'] += int(row['selected_is_guided'] is True)
        if row.get('other_agvs_last_node_nearby', 0) > 0:
            c['decisions_near_other_agvs'] += 1
            bucket['nearby_traffic_decisions'] += 1
        if row.get('remaining_change') is not None:
            if row['remaining_change'] < -1e-6:
                c['towards_target'] += 1
                bucket['towards_target'] += 1
            elif row['remaining_change'] > 1e-6:
                c['away_from_target'] += 1
                bucket['away_from_target'] += 1
            else:
                c['same_remaining_distance'] += 1
        if row['action'] == 0:
            bucket['stops'] += 1
        if row.get('action_edge_valid') is False:
            c['selected_moves_not_directed_arc'] += 1
            bucket['selected_moves_not_directed_arc'] += 1
        if row.get('target_changed_for_physical_slot'):
            c['observed_target_changes'] += 1
        if row.get('current_node') is not None and row.get('next_node') is not None:
            transitions[(row['current_node'], row['next_node'])] += 1
        # Delta is from previous ROUTING observation, not per chosen action.
        # A rise here may be associated with many unlogged AnyLogic events.
        rises = row.get('stats_increase_since_previous_routing_observation') or {}
        for name in ('block', 'liveLock'):
            if rises.get(name, 0) is not None and rises.get(name, 0) > 0:
                c[f'intervals_with_{name}_increase'] += 1
                bucket[f'intervals_with_{name}_increase'] += 1
                bucket[f'{name}_increase_amount'] += float(rises[name])
    decision_count = c['routing_decisions']
    guide_count = max(1, c['with_guide'])
    node_rows = []
    for node, stats in nodes.items():
        item = dict(node=node, **stats)
        item['nearby_traffic_share'] = round(stats['nearby_traffic_decisions']/stats['decisions'], 4)
        item['guide_follow_rate'] = round(stats['guide_followed']/max(1,stats['guide_available']),4)
        node_rows.append(item)
    node_rows.sort(key=lambda r: (-r.get('decisions', 0), r['node']))
    return dict(total_decisions=decision_count,
                stops=c['stops'],
                guide_available=c['with_guide'], guide_followed=c['guide_followed'],
                guide_follow_rate=round(c['guide_followed']/guide_count, 4),
                toward_target=c['towards_target'], away_from_target=c['away_from_target'],
                selected_moves_not_directed_arc=c['selected_moves_not_directed_arc'],
                decisions_near_other_agvs=c['decisions_near_other_agvs'],
                target_changes_observed=c['observed_target_changes'],
                intervals_with_block_increase=c['intervals_with_block_increase'],
                intervals_with_liveLock_increase=c['intervals_with_liveLock_increase'],
                hotspot_nodes=node_rows,
                top_transitions=[{'from':u, 'to':v, 'decisions':count}
                    for (u,v),count in transitions.most_common(20)],
                interpretation=(
                    'Node ranking is by observed routing decision volume, NOT causal hotspot risk. '
                    'Interval blockage/livelock rises are contemporaneous counter changes '
                    'between routing observations, NOT proof a selected action caused them. '
                    'AGV proximity uses last-node positions from actor observation. '
                    'No route probability, safety guarantee or throughput improvement is inferred.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--top', type=int, default=12)
    args = p.parse_args()
    folder = args.folder.resolve()
    files = sorted(folder.glob('routing_trace_seed*.jsonl'))
    if not files:
        p.error('No routing_trace_seed*.jsonl found in '+str(folder))
    if args.top < 1:
        p.error('--top must be positive')
    out = {}
    for file in files:
        seed = file.stem.split('seed')[-1]
        result = summarize(parse_rows(file))
        out[seed] = result
        print(f'\n=== SEED {seed} ===')
        for key in ('total_decisions', 'stops', 'guide_follow_rate',
                    'toward_target', 'away_from_target',
                    'decisions_near_other_agvs',
                    'intervals_with_block_increase',
                    'intervals_with_liveLock_increase',
                    'selected_moves_not_directed_arc'):
            print(f'  {key}: {result[key]}')
        print('Top routing activity nodes (not proven collision locations):')
        for node in result['hotspot_nodes'][:args.top]:
            print(f'  node={node["node"]}: decisions={node["decisions"]} '
                  f'nearby={node.get("nearby_traffic_decisions",0)} '
                  f'block_counter_increase_intervals={node.get("intervals_with_block_increase",0)}')
    # Create ONLY analysis outputs; refuse to silently replace an older report.
    report = folder/'routing_trace_analysis_v3.json'
    if report.exists():
        p.error('Refusing to overwrite existing report: '+str(report))
    report.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSaved:', report)
    print('NOTE: Block/livelock increases cannot be attributed causally to prior actions.','\n')


if __name__ == '__main__':
    main()
