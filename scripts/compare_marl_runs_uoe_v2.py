"""Read-only JSONL comparison inspired by EPyMARL evaluation discipline.

Works with train metrics.jsonl and held-out evaluation*.jsonl. Does NOT
claim a scientifically significant result from a single seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

FIELDS = ('delivered_reward', 'jobDone', 'targetReached', 'wrongTarget',
          'block', 'liveLock', 'closed_transitions')


def metric(row, key):
    env = row.get('environment', {})
    if key in env:
        return env[key]
    return env.get('statistics', {}).get(key)


def summarize(file):
    data = [json.loads(line) for line in Path(file).read_text(encoding='utf8').splitlines() if line.strip()]
    if not data:
        raise ValueError('No rows in '+str(file))
    rows = {}
    for key in FIELDS:
        valid = [float(v) for row in data if (v := metric(row, key)) is not None]
        if valid:
            rows[key] = dict(n=len(valid), mean=round(mean(valid), 4),
                             sd=round(stdev(valid), 4) if len(valid) > 1 else None)
    seeds = [row.get('seed', row.get('environment_seed')) for row in data]
    return data, seeds, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+', type=Path)
    args = p.parse_args()
    for file in args.files:
        data, seeds, results = summarize(file)
        print('\n===', file, '===')
        print('Rows:', len(data), ' Seeds:', seeds)
        print('Evaluation:', all('seed' in row and 'episode' not in row for row in data))
        for k, x in results.items():
            print(f'{k:>22}: mean={x["mean"]}, sd={x["sd"]}, n={x["n"]}')
    if len(args.files) > 1:
        print('\nWARNING: Compare policies on held-out identical seeds, sim-seconds, '+
              'fleet size, model ZIP, and episode termination rules. '+
              'Training rows are NOT controlled evaluation.')


if __name__ == '__main__':
    main()
