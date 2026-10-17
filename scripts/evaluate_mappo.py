"""Evaluate MAPPO checkpoint on Matrix at fixed seeds and fleet sizes.

Uses exactly the same AnyLogic model/reward setup as train_mappo.py.
For a fair baseline, separately run original DQN+PPO with matching seeds/config.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import train_mappo as training
from thesis.mappo import EventCollector


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', default=str(ROOT/'envs'/'Matrix.zip'))
    p.add_argument('--output', default=str(ROOT/'mappo_runs'/'evaluation.csv'))
    p.add_argument('--fleetsizes', nargs='+', type=int, default=[8,10,12,14,16,18,20])
    p.add_argument('--seeds', nargs='+', type=int, default=[43,44,45])
    p.add_argument('--max-fleetsize', type=int, default=20)
    p.add_argument('--sim-seconds', type=int, default=3600)
    p.add_argument('--max-events', type=int, default=20000)
    p.add_argument('--port', type=int, default=51150)
    p.add_argument('--routing-backbone', default='original_gnn')
    p.add_argument('--dispatch-backbone', default='original_linear')
    p.add_argument('--env-type', default='matrix')
    p.add_argument('--gamma', type=float, default=0.90)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--mini-batches', type=int, default=1)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--single-dispatch-policy', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=43)
    args=p.parse_args()
    results=[]
    for fleet in args.fleetsizes:
        args.fleetsize = fleet
        env=None
        try:
            env=training.build_env(args)
            model=training.make_model(env,args)
            model.load(args.checkpoint)
            collector=EventCollector(model,env)
            for seed in args.seeds:
                _, meta=collector.episode(seed=seed,train=False,max_event_steps=args.max_events)
                stats=meta.pop('statistics',{})
                row=dict(fleetsize=fleet,seed=seed,**meta)
                # Preserve AnyLogic's exact KPI/statistic names in output.
                row.update({('stat_'+str(k)):v for k,v in stats.items()})
                results.append(row)
                print(f'{fleet} AGVs, seed {seed}: events={meta["events"]}, terminal={meta["reached_terminal"]}')
        finally:
            if env is not None:
                env.close()
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    columns=sorted(set(k for row in results for k in row))
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print(f'Saved {output}')


if __name__=='__main__':main()
