"""Paired HELD-OUT comparison B2 versus graph-guided Routing B3.

Do not compare training rewards for different environment seeds.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from statistics import mean


def load(path):
    by_seed = {}
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        r = json.loads(line)
        seed = int(r['seed'])
        if seed in by_seed: raise ValueError(f'Duplicate seed {seed} in {path}')
        by_seed[seed] = r
    if not by_seed: raise ValueError(f'No evaluations in {path}')
    return by_seed


def metrics(row):
    env = row['environment']; s = env.get('statistics',{})
    actions = env.get('action_diagnostics',{}).get('actions_by_role',{})
    disp = {}
    for role, counts in actions.items():
        if role.startswith('dispatch'):
            for k, v in counts.items():
                disp[int(k)] = disp.get(int(k), 0)+int(v)
    guide = env.get('routing_guidance',{})
    target = float(s.get('targetReached',0)); wrong = float(s.get('wrongTarget',0))
    ops = sum(float(v) for k,v in s.items() if k.startswith('_jobnum_'))
    return dict(reward=round(float(env.get('delivered_reward',0)),3),
                jobDone=float(s.get('jobDone',0)),
                operations=ops, targetReached=target,wrongTarget=wrong,
                arrival_success_pct=round(100*target/max(1,target+wrong),3),
                invalid=float(s.get('taken_invalid',0)),
                blocks=float(s.get('block',0)), livelocks=float(s.get('liveLock',0)),
                buffer_dispatch=sum(v for a,v in disp.items() if 21 <= a <= 25),
                total_dispatch=sum(disp.values()),
                guided_decisions=int(guide.get('guided',0)),
                guide_hit_pct=round(100*int(guide.get('selected_guide',0))/max(1,int(guide.get('guided_actions',0))),3),
                failed_guidance=int(guide.get('no_legal_path',0)) + int(guide.get('unmapped_coordinates',0)) + int(guide.get('missing_graph',0)),
                )


def compare(b2path,b3path,out):
    b2,b3=load(b2path),load(b3path)
    if set(b2)!=set(b3): raise ValueError(f'Different held-out seeds: B2={sorted(b2)}, B3={sorted(b3)}')
    records = []
    for seed in sorted(b2):
        aa,bb=metrics(b2[seed]),metrics(b3[seed])
        records.extend([dict(seed=seed,model='B2',**aa),dict(seed=seed,model='B3',**bb)])
    dest=Path(out)
    if dest.exists():raise FileExistsError(f'Output exists; no overwrite: {dest}')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print('Paired held-out comparison (higher arrival success/operations/jobDone preferred):')
    keys=['reward','jobDone','operations','targetReached','wrongTarget','arrival_success_pct','blocks','livelocks','guide_hit_pct']
    for key in keys:
        a=mean(float(r[key]) for r in records if r['model']=='B2')
        b=mean(float(r[key]) for r in records if r['model']=='B3')
        print(f'  {key:>20s}: B2={a:>10.3f}, B3={b:>10.3f}, Δ(B3-B2)={b-a:>10.3f}')
    print('Seed-level descriptive comparison only; no statistical significance claim.')
    print('IMPORTANT: Guidance hit can improve by design; look for jobDone, operations and targetReached.')
    print(f'[CSV] {dest}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--b2',required=True)
    parser.add_argument('--b3',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    compare(args.b2,args.b3,args.out)
