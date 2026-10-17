"""Compare completed A/B experiment metrics.jsonl. No external dependencies."""
import argparse
import json
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser(description='Compare MAPPO Stop A/B runs')
parser.add_argument('baseline', type=Path, help='A metrics.jsonl')
parser.add_argument('modified', type=Path, help='B metrics.jsonl')
args = parser.parse_args()

def load(path):
    with path.open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f'No episodes found in {path}')
    return rows

a, b = load(args.baseline), load(args.modified)
if len(a) != len(b):
    print(f'WARNING: Different episode counts A={len(a)}, B={len(b)}')

def stat(row, key):
    return float(row['environment'].get('statistics', {}).get(key, 0))

def diag(row, key):
    return float(row['environment'].get('action_diagnostics', {}).get(key, 0))

def routing_stop_pct(row):
    counts = row['environment'].get('action_diagnostics', {}).get('actions_by_role', {}).get('routing', {})
    total = sum(counts.values())
    return 100.0 * counts.get('0', 0) / total if total else 0

measures = [
    ('Reward delivered', lambda r: float(r['environment']['delivered_reward'])),
    ('jobDone', lambda r: stat(r, 'jobDone')),
    ('targetReached', lambda r: stat(r, 'targetReached')),
    ('wrongTarget', lambda r: stat(r, 'wrongTarget')),
    ('taken_invalid', lambda r: stat(r, 'taken_invalid')),
    ('block', lambda r: stat(r, 'block')),
    ('liveLock', lambda r: stat(r, 'liveLock')),
    ('routing action 0 (%)', routing_stop_pct),
    ('stop with move possible', lambda r: diag(r, 'routing_stop_with_move_available')),
    ('stop masked decisions', lambda r: diag(r, 'routing_stop_masked_decisions')),
    ('mask violations', lambda r: diag(r, 'selected_mask_violations')),
    ('fresh_routing', lambda r: float(r['environment'].get('station_state_sources', {}).get('fresh_routing', 0))),
]
for name, data in [('A', a), ('B', b)]:
    runs = [r.get('experiment', {}) for r in data]
    print(f'{name}: episodes={len(data)}, stop_mask_mode={runs[0].get("stop_mask_mode", "unknown")}, seed={runs[0].get("seed", "unknown")}')
print('\nMeans per episode (B-A is only descriptive, not a significance test):')
print(f'{"Metric":30s} {"A baseline":>14s} {"B restricted":>14s} {"B - A":>14s}')
print('-' * 76)
for label, func in measures:
    x = mean(func(row) for row in a)
    y = mean(func(row) for row in b)
    print(f'{label:30s} {x:14.3f} {y:14.3f} {y-x:14.3f}')
print('\nReminder: this changes the available action set; compare multiple seeds before claiming policy improvement.')
