"""Plot 50-episode MAPPO learning diagnostics from TRAINING metrics.jsonl.

Important: a training reward curve is not an unbiased evaluation curve.
Use evaluate_mappo_long_v121.py on held-out seeds for evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def moving_mean(values, window=5):
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
            for i in range(len(values))]


def summarize_metrics(metrics_file: Path, window: int = 5):
    lines = metrics_file.read_text(encoding='utf-8').splitlines()
    episodes = [json.loads(line) for line in lines if line.strip()]
    if not episodes:
        raise ValueError(f'No episodes found in {metrics_file}')
    rows = []
    for line in episodes:
        env = line.get('environment', {})
        stats = env.get('statistics', {})
        diag = env.get('action_diagnostics', {})
        routing = diag.get('actions_by_role', {}).get('routing', {})
        routing_actions = sum(int(v) for v in routing.values())
        stop_rate = 100 * float(routing.get('0', 0)) / routing_actions if routing_actions else 0
        rows.append({
            'episode': int(line['episode']),
            'environment_seed': int(line.get('environment_seed', 0)),
            'reward_delivered': float(env.get('delivered_reward', 0)),
            'jobDone': float(stats.get('jobDone', 0)),
            'targetReached': float(stats.get('targetReached', 0)),
            'wrongTarget': float(stats.get('wrongTarget', 0)),
            'taken_invalid': float(stats.get('taken_invalid', 0)),
            'block': float(stats.get('block', 0)),
            'liveLock': float(stats.get('liveLock', 0)),
            'newPart': float(stats.get('newPart', 0)),
            'operations': float(sum(v for k, v in stats.items() if k.startswith('_jobnum_') and isinstance(v, (int, float)))),
            'routing_action0_pct': round(stop_rate, 3),
            'transitions': int(env.get('closed_transitions', 0)),
            'fresh_routing': int(env.get('station_state_sources', {}).get('fresh_routing', 0)),
            'routing_updates': int(line.get('policy_updates', {}).get('routing', {}).get('transitions', 0)),
            'dispatch_front_updates': int(line.get('policy_updates', {}).get('dispatch_front', {}).get('transitions', 0)),
            'dispatch_rear_updates': int(line.get('policy_updates', {}).get('dispatch_rear', {}).get('transitions', 0)),
        })
    for key in ('reward_delivered','jobDone','targetReached','wrongTarget','taken_invalid','block','operations'):
        series = moving_mean([row[key] for row in rows], window)
        for row, val in zip(rows, series):
            row[f'{key}_rolling{window}'] = round(val, 4)
    return rows


def save_report(rows, output_dir: Path, window: int = 5, make_plots: bool = True):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'learning_progress.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = min(10, len(rows))
    begin = rows[:count]
    end = rows[-count:]
    compare = ['reward_delivered','jobDone','targetReached','wrongTarget','taken_invalid','block','operations']
    report = [f'Training episodes recorded: {len(rows)}',
              f'Comparison: first {count} vs last {count} TRAINING episodes (descriptive only).']
    for key in compare:
        a=sum(row[key] for row in begin)/count
        b=sum(row[key] for row in end)/count
        report.append(f'{key:22s} first={a:10.3f}  last={b:10.3f}  delta={b-a:+10.3f}')
    report.append('Note: evaluate checkpoint on held-out seeds before claiming policy improvement.')
    (output_dir / 'learning_summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    if make_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for field, title in (
            ('reward_delivered','Delivered reward'),
            ('jobDone','Completed products (jobDone)'),
            ('targetReached','AGV targetReached'),
            ('wrongTarget','AGV wrongTarget'),
            ('taken_invalid','Invalid actions'),
        ):
            fig, ax = plt.subplots(figsize=(9, 4.3))
            x = [r['episode'] for r in rows]
            ax.plot(x, [r[field] for r in rows], alpha=.5, label='Each episode')
            ax.plot(x, [r[f'{field}_rolling{window}'] for r in rows], label=f'Moving avg ({window})')
            ax.set(title=f'MAPPO A (training): {title}', xlabel='Episode', ylabel=title)
            ax.legend(loc='best')
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(output_dir / f'learning_{field}.png', dpi=140)
            plt.close(fig)
    return csv_path, output_dir / 'learning_summary.txt'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('metrics', type=Path, nargs='?', default=Path('mappo_runs/long_A_seed43_50ep/metrics.jsonl'))
    p.add_argument('--window', type=int, default=5)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--no-plots', action='store_true')
    args=p.parse_args()
    if args.window <= 0: p.error('--window must be positive')
    rows=summarize_metrics(args.metrics, args.window)
    out=args.output or args.metrics.parent
    report_path, summary_path=save_report(rows, out, args.window, not args.no_plots)
    print(summary_path.read_text(encoding='utf-8'))
    print(f'[FILES] {report_path} and PNG learning curves in {out}')


if __name__=='__main__':
    main()
