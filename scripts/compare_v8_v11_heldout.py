"""So sanh held-out V8/V11, cung seed. Chi doc file, khong can AnyLogic."""
import argparse
import csv
import json
from pathlib import Path

FIELDS = ('jobDone','wrongTarget','block','liveLock','block_time_s','targetReached','reward')

def row_metrics(row):
    md = row.get('environment', {})
    stats = md.get('statistics', {})
    return {
        'jobDone': float(stats.get('jobDone', 0)),
        'wrongTarget': float(stats.get('wrongTarget', 0)),
        'block': float(stats.get('block', 0)),
        'liveLock': float(stats.get('liveLock', 0)),
        'block_time_s': sum(float(v) for k, v in stats.items() if k.startswith('_blocksec_agv')),
        'targetReached': float(stats.get('targetReached', 0)),
        'reward': float(md.get('delivered_reward', 0)),
    }

def read_file(p):
    records = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not records:
        raise ValueError(f'Empty: {p}')
    unique = {int(r['seed']): r for r in records}
    if len(unique) != len(records):
        raise ValueError(f'Duplicate seeds: {p}')
    return {seed: row_metrics(r) for seed, r in unique.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args()
    files=sorted(args.directory.glob('heldout_*_5seeds.jsonl'))
    if not files:
        raise SystemExit('No heldout files to compare')
    all_results={f.stem: read_file(f) for f in files}
    seedsets={tuple(sorted(d)) for d in all_results.values()}
    if len(seedsets)>1:
        raise SystemExit('Evaluation seed sets DIFFER: do not compare until consistent')
    summary=[]
    for name,data in all_results.items():
        line={'checkpoint':name,'n_seeds':len(data)}
        for key in FIELDS:
            line[key]=sum(d[key] for d in data.values())/len(data)
        summary.append(line)
    out=args.directory/'heldout_comparison.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['checkpoint','n_seeds']+list(FIELDS))
        w.writeheader();w.writerows(summary)
    print('\n=== HELD-OUT: SAME 5 SEEDS (higher JobDone/Reward, lower other costs) ===')
    print(f"{'Checkpoint':<38} {'JobDone':>8} {'Wrong':>8} {'Block':>8} {'LiveLock':>9} {'BlockSec':>10} {'Reward':>10}")
    for d in summary:
        print(f"{d['checkpoint']:<38} {d['jobDone']:>8.2f} {d['wrongTarget']:>8.2f} "
              f"{d['block']:>8.2f} {d['liveLock']:>9.2f} {d['block_time_s']:>10.1f} {d['reward']:>10.1f}")
    print(f'\nSaved CSV: {out}')
    print('Luu y: 5 seeds de tham do; chua du de khang dinh thong ke manh.')

if __name__=='__main__': main()
