"""Analyze COMPLETE A/B/C/D factorial experiment with matched training/eval seeds.

Reports both per-evaluation rows and per-training-seed averages. Pairing is strict.
No significance claims from 1 training seed, even with 3 eval seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ablation_abcd_v5_common import ARMS, METRICS, run_dir, assert_distinct_seeds


def read_records(folder: Path) -> tuple[dict, list[dict]]:
    manifestfile = folder / 'experiment_manifest.json'
    if not manifestfile.is_file():
        raise FileNotFoundError('Experiment manifest missing: ' + str(manifestfile))
    config = json.loads(manifestfile.read_text(encoding='utf-8'))
    train_seeds = config['train_seeds']
    eval_seeds = config['eval_seeds']
    assert_distinct_seeds(train_seeds, eval_seeds, config['train_episodes'])
    records = []
    graph_fingerprints = set()
    comparable_configs = set()
    for arm in ARMS:
        for ts in train_seeds:
            d = run_dir(folder, arm, ts)
            cf = d / 'run_config.json'
            f = d / f'evaluation_{config["eval_sim_seconds"]}.jsonl'
            if not (cf.is_file() and f.is_file()):
                raise FileNotFoundError(f'Missing training config or evaluation: {d}')
            run = json.loads(cf.read_text(encoding='utf-8'))
            graph_fingerprints.add(run['graph_fingerprint'])
            # Strictly compare shared experiment hyperparameters, excluding encoder/guide.
            shared = ('model_file','max_events','fleetsize','max_fleetsize',
                      'single_dispatch_policy','stop_mask_mode','graph_layers',
                      'hidden','gamma','gae_lambda','clip','lr','epochs','mini_batches')
            comparable_configs.add(tuple((key,str(run.get(key))) for key in shared))
            if (run['seed'] != ts or run['episodes'] != config['train_episodes']
                or run['sim_seconds'] != config['train_sim_seconds']
                or run.get('dispatch_backbone') != 'pooled'
                or run.get('dispatch_mask_mode') != 'operation_first'):
                raise RuntimeError(f'Config mismatch: {cf}')
            if arm.encoder == 'gat':
                strength = float(run['routing_guide_strength'])
                if run['routing_encoder'] != 'gat':
                    raise RuntimeError(f'Wrong encoder {cf}')
            else:
                strength = float(run['guide_start'])
                if (run['routing_encoder'] != 'residual_edge_gat' or
                    not math.isclose(strength, float(run['guide_final']), abs_tol=1e-12)):
                    raise RuntimeError(f'Wrong encoder or nonconstant guide: {cf}')
            if not math.isclose(strength, arm.guide, abs_tol=1e-12):
                raise RuntimeError(f'Wrong Dijkstra strength: {cf}')
            rows = [json.loads(s) for s in f.read_text(encoding='utf-8').splitlines() if s.strip()]
            if len(rows) != len(eval_seeds) or [int(r['seed']) for r in rows] != eval_seeds:
                raise RuntimeError(f'Missing/duplicate/unordered eval seeds in {f}')
            for row in rows:
                meta = row['environment']
                if arm.encoder == 'edge_gat':
                    if not math.isclose(float(meta.get('dijkstra_strength_eval', -1)),
                                        arm.guide, abs_tol=1e-12):
                        raise RuntimeError(f'Wrong evaluation-time guidance: {f}')
                stats = meta.get('statistics', {})
                record = {'arm': arm.code, 'encoder': arm.encoder, 'guide': arm.guide,
                          'train_seed': ts, 'eval_seed': int(row['seed'])}
                if not meta.get('reached_terminal', False):
                    raise RuntimeError(f'Environment did not reach terminal: {f} seed {row["seed"]}')
                for m in METRICS:
                    val = meta.get('delivered_reward') if m == 'delivered_reward' else stats.get(m)
                    if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                        raise RuntimeError(f'Missing/invalid {m} in {f} seed {row["seed"]}: {val}')
                    record[m] = float(val)
                records.append(record)
    if len(comparable_configs) != 1:
        raise RuntimeError('Nonfactor experimental settings differ among arms/seeds')
    if len(graph_fingerprints) != 1:
        raise RuntimeError(f'Graph topology mismatch across runs: {graph_fingerprints}')
    return config, records


def mean(x):
    return statistics.mean(x)


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else None


def calculate(config: dict, records: list[dict]):
    byarm = defaultdict(list)
    matched = {}
    for row in records:
        byarm[row['arm']].append(row)
        key = (row['train_seed'], row['eval_seed'])
        matched.setdefault(key, {})[row['arm']] = row
    for pair, arm_values in matched.items():
        if set(arm_values) != {'A', 'B', 'C', 'D'}:
            raise RuntimeError(f'Incomplete matched pair: {pair}, present: {sorted(arm_values)}')
    summary = []
    for arm in ARMS:
        arr = byarm[arm.code]
        per_train = []
        for ts in config['train_seeds']:
            a = [r for r in arr if r['train_seed'] == ts]
            if len(a) != len(config['eval_seeds']):
                raise RuntimeError(f'Incomplete train seed: {arm.code} {ts}')
            per_train.append({m: mean([r[m] for r in a]) for m in METRICS})
        for metric in METRICS:
            seed_means = [x[metric] for x in per_train]
            summary.append(dict(arm=arm.code, encoder=arm.encoder, guide=arm.guide,
                                metric=metric, mean_over_training_seeds=mean(seed_means),
                                sd_between_training_seeds=sd(seed_means),
                                n_training_seeds=len(seed_means),
                                n_eval_seeds_per_training=len(config['eval_seeds']),
                                pooled_eval_sd_descriptive=sd([r[metric] for r in arr])))
    effects = []
    for metric in METRICS:
        for name, weights in (
            ('EdgeGAT_vs_GAT', {'A': -.5, 'B': -.5, 'C': .5, 'D': .5}),
            ('Guide0p5_vs_1p5', {'A': -.5, 'B': .5, 'C': -.5, 'D': .5}),
            ('Interaction_difference_in_differences', {'A': 1, 'B': -1, 'C': -1, 'D': 1}),
        ):
            individual = []
            # Use training seed as replication unit, *not* independent eval episodes.
            for ts in config['train_seeds']:
                pairs = [pair for pair in matched if pair[0] == ts]
                individual.append(mean([sum(weights[k] * matched[pair][k][metric]
                                           for k in weights) for pair in pairs]))
            effects.append(dict(metric=metric, contrast=name,
                                average=mean(individual), sd_between_training_seeds=sd(individual),
                                n_training_seeds=len(individual)))
    return summary, effects


def write_csv(path, rows):
    if not rows:
        raise RuntimeError('No rows to write')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def report(folder: Path):
    cfg, rows = read_records(folder)
    summary, contrasts = calculate(cfg, rows)
    write_csv(folder/'all_evaluations.csv', rows)
    write_csv(folder/'condition_summary.csv', summary)
    write_csv(folder/'factorial_contrasts.csv', contrasts)
    table = {(r['arm'], r['metric']): r['mean_over_training_seeds'] for r in summary}
    lines = [
        '# Ablation A/B/C/D — MAPPO Routing', '',
        f"**Training seeds**: {cfg['train_seeds']}; **evaluation seeds**: {cfg['eval_seeds']}",
        f"Train: {cfg['train_episodes']} episodes × {cfg['train_sim_seconds']} simulated seconds; "
        f"Eval: {cfg['eval_sim_seconds']} seconds/seed.", '',
        '| Metric | A GAT,1.5 | B GAT,0.5 | C EdgeGAT,1.5 | D EdgeGAT,0.5 |',
        '|---|---:|---:|---:|---:|',
    ]
    for m in METRICS:
        lines.append('| ' + m + ' | ' + ' | '.join(f'{table[arm, m]:.3f}' for arm in 'ABCD') + ' |')
    lines += ['', '## Factorial effects (matched training seeds; descriptive)',
              '', '| Metric | EdgeGAT − GAT | Guide 0.5 − 1.5 | Interaction |',
              '|---|---:|---:|---:|']
    effects = {(r['metric'], r['contrast']): r['average'] for r in contrasts}
    for m in METRICS:
        lines.append(f'| {m} | {effects[m,"EdgeGAT_vs_GAT"]:.3f} | '
                     f'{effects[m,"Guide0p5_vs_1p5"]:.3f} | '
                     f'{effects[m,"Interaction_difference_in_differences"]:.3f} |')
    lines += ['', '## Interpretation',
              '', '- Higher **jobDone**, **targetReached**: desirable.',
              '- Lower **block**, **liveLock**, **wrongTarget**: desirable.',
              '- The *interaction* equals (D − C) − (B − A); positive is not universally better.',
              '- Every A/B/C/D arm is **trained from scratch** using its own fixed guidance strength.',
              '- Checkpoints are not swapped between architectures; evaluation does not train.',
              '- An experiment with only one training seed is a **pilot**, not a reliable significance test.',
              '- No causal claim for congestion source or edge-GAT advantage without repeated training seeds.',
              '- Graph feature units are not necessarily meters; avoid any unsupported physical interpretation.',
              '']
    out = folder/'ABCD_REPORT.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[:lines.index('## Interpretation')]))
    print('\nSaved:', out, folder/'all_evaluations.csv',
          folder/'condition_summary.csv', folder/'factorial_contrasts.csv')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, default=Path('mappo_runs/B3_ABCD_V5_pilot'))
    a = p.parse_args()
    report(a.folder.resolve())


if __name__ == '__main__':
    main()
