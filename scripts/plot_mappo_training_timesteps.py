#!/usr/bin/env python3
"""6-panel paper-style training curves against *recorded* on-policy transitions.

Each recorded ``environment.closed_transitions`` contributes that episode's
number of finished RL transitions. This is NOT simulation seconds and is not
assumed equivalent to the exact "Training Steps" counter of Steinbacher et al.

Examples:
  python scripts/plot_mappo_training_timesteps.py --folders mappo_runs/V8_15AGV_fromV7_5h
  python scripts/plot_mappo_training_timesteps.py --folders mappo_runs/V8_15AGV_fromV7_5h mappo_runs/V8_20AGV_from15_5h
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = (
    ('reward', 'Episode Reward', 'Reward'),
    ('targetReached', 'Targets Reached', 'Targets Reached'),
    ('block', 'Number of Blocks', 'Blocks'),
    ('jobDone', 'Products Completed', 'Products'),
    ('wrongTarget', 'Wrong Targets', 'Wrong Targets'),
    ('liveLock', 'Livelock', 'Livelock'),
)


def numerical(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def episode_rows(folder: Path):
    path = folder/'metrics.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'No recorded training data: {path}')
    result = []
    with path.open('r', encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle,1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f'Invalid JSONL at {path}:{line_number}: {err}') from err
            env = record.get('environment')
            if not isinstance(env,dict):
                raise ValueError(f'{path}:{line_number} missing environment dictionary')
            steps = numerical(env.get('closed_transitions'))
            if steps is None or steps <= 0 or int(steps) != steps:
                raise ValueError(f'{path}:{line_number} no valid environment.closed_transitions. '
                                 'Cannot create honest training timestep axis.')
            episode = numerical(record.get('episode'))
            if episode is None or int(episode) != episode:
                raise ValueError(f'{path}:{line_number} no valid episode field')
            stats = env.get('statistics') or {}
            if not isinstance(stats,dict):
                stats = {}
            data = {
                'episode': int(episode),
                'transitions': int(steps),
                'reward': numerical(env.get('delivered_reward')),
                'targetReached': numerical(stats.get('targetReached')),
                'block': numerical(stats.get('block')),
                'jobDone': numerical(stats.get('jobDone')),
                'wrongTarget': numerical(stats.get('wrongTarget')),
                'liveLock': numerical(stats.get('liveLock')),
            }
            result.append(data)
    if not result:
        raise ValueError(f'{path} contains no episode records')
    result.sort(key=lambda r: r['episode'])
    if len(set(r['episode'] for r in result)) != len(result):
        raise ValueError(f'{path} has duplicate episode records')
    return result


def load_training_phases(folders):
    """Carry timesteps across adjacent phases, never invent the missing V7 steps."""
    cumulative = 0
    last_episode = -1
    result = []
    for folder in folders:
        rows = episode_rows(folder)
        if rows[0]['episode'] <= last_episode:
            raise ValueError(f'Episodes overlap or run folders are out of order: {folder}')
        # Actual fleet comes from run config, not the folder name.
        config_file = folder/'run_config.json'
        if not config_file.is_file():
            raise FileNotFoundError(f'Missing run_config.json needed to label fleetsize: {config_file}')
        config = json.loads(config_file.read_text(encoding='utf-8-sig'))
        fleet = config.get('fleetsize')
        if not isinstance(fleet,int) or fleet < 1:
            raise ValueError(f'{config_file} has no valid fleetsize')
        for row in rows:
            cumulative += row['transitions']
            row['cumulative_transitions'] = cumulative
            row['fleetsize'] = fleet
            row['source_run'] = folder.name
        result.append((folder, fleet, rows))
        last_episode = rows[-1]['episode']
    return result


def moving_average(values, w):
    result = []
    for i in range(len(values)):
        seq=[x for x in values[max(0,i-w+1):i+1] if x is not None]
        result.append(sum(seq)/len(seq) if seq else float('nan'))
    return result


def plot(phases, output, window):
    import matplotlib.ticker as ticker
    fig, axs = plt.subplots(3,2,figsize=(13.5,12),constrained_layout=True)
    colors=('#ba2149','#5446a3','#207f76','#cd8c10')
    for ax, (key,title,ylab) in zip(axs.ravel(),PANELS):
        for idx,(folder,fleet,rows) in enumerate(phases):
            relevant=[r for r in rows if r[key] is not None]
            if not relevant:
                continue
            x=[r['cumulative_transitions'] for r in relevant]
            y=[r[key] for r in relevant]
            color=colors[idx%len(colors)]
            ax.plot(x,y,alpha=.20,lw=.8,color=color)
            ax.plot(x,moving_average(y,window),lw=1.9,color=color,
                    label=f'{fleet} AGVs — rolling mean ({window})')
        for folder,fleet,rows in phases[:-1]:
            ax.axvline(rows[-1]['cumulative_transitions'],color='#444444',lw=.9,ls='--',alpha=.75)
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('Training timesteps (cumulative closed RL transitions)')
        ax.set_ylabel(ylab)
        ax.grid(axis='y',alpha=.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.ticklabel_format(style='sci',axis='x',scilimits=(4,4))
        ax.legend(fontsize=8,loc='best')
    fig.suptitle('MAPPO training — measured transitions (not simulation seconds)',fontsize=13)
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):
        dst=output.with_suffix(suffix)
        fig.savefig(dst,dpi=240,bbox_inches='tight')
        print('[SAVED]',dst)
    plt.close(fig)
    csvfile=output.parent/'training_timesteps_data.csv'
    fields=['source_run','fleetsize','episode','transitions','cumulative_transitions']+[p[0] for p in PANELS]
    with csvfile.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for _,_,rows in phases:
            writer.writerows(rows)
    print('[SAVED]',csvfile)
    print('NOTE: cumulative counts start at first logged episode in first listed run.')
    print('NOTE: author paper "Training Steps" may use a different step definition; verify before comparing x-axis magnitudes.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folders',nargs='+',type=Path,required=True,
                        help='Chronological run folders (e.g., V8 15AGV then V8 20AGV)')
    parser.add_argument('--window',type=int,default=7,help='Moving average window per fleet phase')
    parser.add_argument('--out',type=Path,default=Path('figures/V8_15to20AGV/Figure_TrainingTimesteps_6panels'))
    args=parser.parse_args(argv)
    if args.window < 1:
        parser.error('--window must be >= 1')
    phases=load_training_phases(args.folders)
    for folder,fleet,rows in phases:
        print(f'[SOURCE] {folder}: {fleet} AGVs, episodes {rows[0]["episode"]}..{rows[-1]["episode"]}, '
              f'last recorded timestep {rows[-1]["cumulative_transitions"]}')
    plot(phases,args.out,args.window)

if __name__=='__main__':
    main()
