"""Paired, held-out evaluation comparison for B vs B2; no training.

Usage:
 python scripts/compare_mappo_B_B2_v121.py --b mappo_runs/long_B_seed43_50ep/evaluation_3600.jsonl --b2 mappo_runs/long_B2_seed43_50ep/evaluation_3600.jsonl
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

BUFFER_ACTIONS = {'21', '22', '23', '24', '25'}


def read(path):
    result={}
    for i, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():
            continue
        row=json.loads(line)
        seed=int(row['seed'])
        if seed in result:raise ValueError(f'{path}:{i}: duplicated seed {seed}')
        meta=row['environment']
        stat=meta.get('statistics',{})
        actions=meta.get('action_diagnostics',{}).get('actions_by_role',{})
        routing=actions.get('routing',{})
        dispatch=[x for name,acts in actions.items() if name.startswith('dispatch') for x,n in acts.items() for _ in range(int(n))]
        buffer_n=sum(a in BUFFER_ACTIONS for a in dispatch)
        reached=float(stat.get('targetReached',0))
        wrong=float(stat.get('wrongTarget',0))
        moves=sum(int(v) for k,v in routing.items() if k!='0')
        stops=int(routing.get('0',0))
        result[seed]={
            'seed':seed,
            'reward':round(float(meta.get('delivered_reward',0)),3),
            'jobDone':stat.get('jobDone',0),
            'operations':sum(float(n) for key,n in stat.items() if key.startswith('_jobnum_')),
            'targetReached':reached,
            'wrongTarget':wrong,
            'wrong_fraction_of_arrivals':round(wrong/(reached+wrong),4) if reached+wrong else None,
            'invalid_per_routing_decision':round(float(stat.get('taken_invalid',0))/(moves+stops),4) if moves+stops else None,
            'block':stat.get('block',0),
            'liveLock':stat.get('liveLock',0),
            'buffer_fraction_of_dispatch':round(buffer_n/len(dispatch),4) if dispatch else None,
            'dispatch_decisions':len(dispatch),
            'routing_decisions':moves+stops,
        }
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--b',type=Path,required=True)
    ap.add_argument('--b2',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    a,b=read(args.b),read(args.b2)
    if not a or a.keys()!=b.keys():
        ap.error(f'Paired seeds must match; B={sorted(a)}, B2={sorted(b)}')
    fields=list(a[next(iter(a))].keys())
    rows=[]
    for seed in sorted(a):
        for variant,data in [('B',a[seed]),('B2',b[seed])]:
            rows.append({'variant':variant,**data})
    print('seed variant jobDone operations targetReached wrongTarget wrongFraction blocks liveLock bufferFraction reward')
    for r in rows:
        print(f"{r['seed']:4d} {r['variant']:>7s} {r['jobDone']:7.0f} {r['operations']:10.0f} {r['targetReached']:13.0f} {r['wrongTarget']:11.0f} {str(r['wrong_fraction_of_arrivals']):>13} {r['block']:6.0f} {r['liveLock']:8.0f} {str(r['buffer_fraction_of_dispatch']):>14} {r['reward']:9.2f}")
    if args.out:
        if args.out.exists():
            ap.error(f'Will not overwrite: {args.out}')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('w',newline='',encoding='utf-8-sig') as fp:
            w=csv.DictWriter(fp,fieldnames=['variant']+fields)
            w.writeheader();w.writerows(rows)
        print(f'[SAVED] {args.out}')
    print('NOTE: B2 buffer suppression is imposed by mask, not evidence of learned avoidance.')

if __name__=='__main__':main()
