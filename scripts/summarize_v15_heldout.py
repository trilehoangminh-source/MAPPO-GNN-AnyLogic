"""Summarize V15 held-out JSONL; descriptive only, not matched Tobias statistical parity."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('file',type=Path)
    args=p.parse_args(argv)
    rows=[json.loads(s) for s in args.file.read_text(encoding='utf8').splitlines() if s.strip()]
    if not rows:raise RuntimeError('No completed evaluation rows')
    if len({r['seed'] for r in rows})!=len(rows):raise RuntimeError('Duplicate evaluation seed')
    if len({(r['phase'],r['checkpoint_sha256'],r['fleet']) for r in rows})!=1:
        raise RuntimeError('Mixed phases/checkpoints/fleet size')
    fields={'JobDone':'jobDone','Block':'block','LiveLock':'liveLock',
            'TargetReached':'targetReached','WrongTarget':'wrongTarget'}
    print('Phase:',rows[0]['phase'],'fleet:',rows[0]['fleet'],'seeds:',len(rows))
    for name,key in fields.items():
        values=[float(r['environment']['statistics'][key]) for r in rows
                if key in r.get('environment',{}).get('statistics',{})]
        if not values:continue
        print(f'{name:15s} mean={np.mean(values):9.2f} std={np.std(values,ddof=1) if len(values)>1 else 0:8.2f} n={len(values)}')
    print('Note: comparison to Tobias Figure 8 requires matched simulation/evaluation definitions.')

if __name__=='__main__':main()
