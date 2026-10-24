"""Read-only V8 overnight run status. Does not require PyTorch/AnyLogic."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, default=Path('mappo_runs/V8_overnight_fromV7_5h'))
    a = p.parse_args()
    f = a.folder
    if not (f/'state.json').exists():
        raise SystemExit('No state.json. Training not started or wrong folder: '+str(f))
    s = json.loads((f/'state.json').read_text(encoding='utf-8'))
    print('V8 last fully saved episode:',s['last_episode'])
    print('Resume checkpoint:', s['checkpoint'])
    print('Saved weight exists:',Path(s['checkpoint']).is_file())
    for name in ['best_mappo.pt','latest_mappo.pt','best_info.json','eval_rankings.json']:
        print(f'{name}:', (f/name).is_file())
    if (f/'best_info.json').is_file():
        print('BEST:', json.loads((f/'best_info.json').read_text(encoding='utf-8')))
    if (f/'metrics.jsonl').exists():
        lines = [x for x in (f/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        if lines:
            last = json.loads(lines[-1]); stats=last.get('environment',{}).get('statistics',{})
            print('Last training: reward',last.get('environment',{}).get('delivered_reward'),
                  'jobDone', stats.get('jobDone'),'block',stats.get('block'),
                  'wrongTarget',stats.get('wrongTarget'))

if __name__ == '__main__': main()
