"""Compare a V2 evaluation with V3.1 trace-enabled evaluation on matching seeds.

Differences indicate replay inconsistency, not necessarily tracer causality:
AnyLogic timing, JVM nondeterminism or simulator RNG may also differ.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

FIELDS = ('delivered_reward', 'jobDone', 'targetReached', 'wrongTarget', 'block', 'liveLock')


def load(path):
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        s = x.get('seed')
        if s is None or s in out:
            raise ValueError('Expected unique seed in '+str(path))
        e = x.get('environment', {})
        stats = e.get('statistics', {})
        out[s] = {k: (e.get(k) if k == 'delivered_reward' else stats.get(k)) for k in FIELDS}
    return out


def compare(a, b, tol=1e-6):
    if set(a) != set(b):
        raise ValueError('Different evaluation seed sets')
    results = []
    for seed in sorted(a):
        for field in FIELDS:
            x, y = a[seed].get(field), b[seed].get(field)
            if x is None or y is None:
                raise ValueError(f'Missing {field} in seed {seed}')
            delta = float(y)-float(x)
            results.append((seed, field, float(x), float(y), delta, abs(delta) <= tol))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v2', required=True, type=Path)
    p.add_argument('--v3', required=True, type=Path)
    p.add_argument('--tolerance', type=float, default=1e-6)
    args = p.parse_args()
    if args.tolerance < 0:
        p.error('Tolerance must be >= 0')
    results = compare(load(args.v2), load(args.v3), args.tolerance)
    for seed, key, v2, v3, delta, ok in results:
        print(f'seed={seed} {key}: V2={v2:g}, V3={v3:g}, delta={delta:g} '
              f'{"MATCH" if ok else "DIFFERENT"}')
    total = sum(int(not r[-1]) for r in results)
    print(f'Noninvasive replay check: {len(results)-total}/{len(results)} fields match')
    if total:
        print('WARNING: Different simulation statistics. Check seed, model, '
              'checkpoint, sim-seconds, model stochasticity and whether '
              'the trace introduced an unexpected effect.')
        raise SystemExit(1)
    print('PASS: these aggregate KPI matched; this does not prove every action was identical.')


if __name__ == '__main__':
    main()
