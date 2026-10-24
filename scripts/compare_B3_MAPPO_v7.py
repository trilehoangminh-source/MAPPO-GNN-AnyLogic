"""Compare paired MAPPO V7 baseline vs stable evaluation, same seed and horizon."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean

KPI = ('jobDone','targetReached','wrongTarget','block','liveLock')


def load(path):
    rows={}
    with Path(path).open('r',encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            x=json.loads(line)
            seed=int(x['seed'])
            if seed in rows:
                raise ValueError('Duplicate evaluation seed: '+str(seed))
            stats=x['environment']['statistics']
            rows[seed]={**{k:float(stats.get(k,0.)) for k in KPI},
                        'reward':float(x['environment']['delivered_reward'])}
    if not rows:
        raise ValueError('No evaluation rows: '+str(path))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--stable',type=Path,required=True)
    args=p.parse_args()
    a=load(args.baseline);b=load(args.stable)
    if a.keys()!=b.keys():
        p.error('Paired evaluation must use identical seeds')
    seeds=sorted(a)
    print('Paired evaluation seeds:',seeds)
    print('KPI                        Baseline       Stable       Delta')
    for k in (*KPI,'reward'):
        av=mean(a[s][k] for s in seeds)
        bv=mean(b[s][k] for s in seeds)
        print(f'{k:22s} {av:12.3f} {bv:12.3f} {bv-av:+12.3f}')
    print('\nThese are EVALUATION seeds; use independent TRAINING seeds for statistical replication.')


if __name__=='__main__':
    main()
