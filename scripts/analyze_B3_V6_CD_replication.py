"""Strict matched C/D comparison. Independent replication unit = TRAINING seed.

Requires all 2 x n_training_seeds x n_evaluation_seeds outcomes.
Outputs per-evaluation data, per-training-seed paired differences and README-style report.
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
from v6_cd_common import CONDITIONS, METRICS, condition_dir, seed_validation


def load_jsonl(filename):
    return [json.loads(s) for s in filename.read_text(encoding='utf-8').splitlines() if s.strip()]


def collect(folder: Path):
    file = folder / 'experiment_manifest.json'
    if not file.is_file():
        raise FileNotFoundError(f'Manifest missing: {file}')
    m = json.loads(file.read_text(encoding='utf-8'))
    if m.get('protocol') != 'V6_CD_EdgeGAT_constant_guide_v1':
        raise RuntimeError('Wrong experiment manifest/protocol')
    if m.get('conditions') != [{'code': c.code, 'encoder': 'residual_edge_gat', 'guide': c.guide} for c in CONDITIONS]:
        raise RuntimeError('Experiment factor configuration mismatch')
    seed_validation(m['train_seeds'], m['eval_seeds'], m['train_episodes'])
    records, graph_ids, shared_params = [], set(), set()
    for c in CONDITIONS:
        for ts in m['train_seeds']:
            d = condition_dir(folder, c, ts)
            config_path = d / 'run_config.json'
            checkpoint = d / f'checkpoint_ep_{m["train_episodes"]:04d}.pt'
            eval_path = d / f'evaluation_{m["eval_sim_seconds"]}.jsonl'
            if not all(f.is_file() for f in (config_path, checkpoint, eval_path)):
                raise FileNotFoundError(f'Missing config/checkpoint/eval: {d}')
            conf = json.loads(config_path.read_text(encoding='utf-8'))
            if (conf.get('version') != 'B3_EDGE_GAT_MAPPO_V4' or
                conf.get('routing_encoder') != 'residual_edge_gat' or
                conf.get('dispatch_backbone') != 'pooled' or
                conf.get('dispatch_mask_mode') != 'operation_first' or
                conf.get('seed') != ts or conf.get('episodes') != m['train_episodes'] or
                conf.get('sim_seconds') != m['train_sim_seconds'] or
                conf.get('max_events') != m['max_events'] or
                conf.get('hidden') != m['hidden'] or
                conf.get('graph_layers') != m['graph_layers'] or
                conf.get('heads') != m['heads'] or conf.get('epochs') != m['epochs'] or
                not math.isclose(float(conf.get('guide_start', -1)), c.guide, abs_tol=1e-10) or
                not math.isclose(float(conf.get('guide_final', -1)), c.guide, abs_tol=1e-10)):
                raise RuntimeError(f'Incompatible training config: {config_path}')
            graph_ids.add(conf.get('graph_fingerprint'))
            # All non-factor hyperparameters must be the same across ALL runs.
            shared = ('model_file', 'max_events', 'fleetsize', 'max_fleetsize',
                      'single_dispatch_policy', 'stop_mask_mode', 'graph_layers', 'hidden',
                      'heads', 'gamma', 'gae_lambda', 'clip', 'lr', 'epochs', 'mini_batches')
            shared_params.add(tuple((k, str(conf.get(k))) for k in shared))
            entries = load_jsonl(eval_path)
            if len(entries) != len(m['eval_seeds']) or [r.get('seed') for r in entries] != m['eval_seeds']:
                raise RuntimeError(f'Missing, duplicate, or unordered evaluation seeds: {eval_path}')
            for rec in entries:
                env = rec.get('environment', {})
                stats = env.get('statistics', {})
                if (rec.get('evaluation') is not True or rec.get('version') != 'B3_EDGE_GAT_MAPPO_V4' or
                    not env.get('reached_terminal', False) or
                    not math.isclose(float(env.get('dijkstra_strength_eval', -1)), c.guide, abs_tol=1e-10) or
                    Path(rec.get('checkpoint', '')).resolve() != checkpoint.resolve()):
                    raise RuntimeError(f'Invalid evaluation record: {eval_path} seed {rec.get("seed")}')
                item = dict(condition=c.code, guide=c.guide, encoder='residual_edge_gat',
                            train_seed=ts, eval_seed=rec['seed'])
                for metric in METRICS:
                    val = env.get('delivered_reward') if metric == 'delivered_reward' else stats.get(metric)
                    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                        raise RuntimeError(f'Missing or nonfinite {metric}: {eval_path} seed {rec["seed"]}')
                    item[metric] = float(val)
                records.append(item)
    if len(graph_ids) != 1 or None in graph_ids:
        raise RuntimeError(f'Graph topology fingerprints differ: {graph_ids}')
    if len(shared_params) != 1:
        raise RuntimeError('Non-factor settings changed across runs')
    return m, records


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else None


def summarize(manifest, records):
    paired = defaultdict(dict)
    for r in records:
        key = (r['train_seed'], r['eval_seed'])
        if r['condition'] in paired[key]:
            raise RuntimeError(f'Duplicate paired observation {key}, {r["condition"]}')
        paired[key][r['condition']] = r
    expected = {(t, s) for t in manifest['train_seeds'] for s in manifest['eval_seeds']}
    if set(paired) != expected or any(set(paired[x]) != {'C', 'D'} for x in paired):
        raise RuntimeError('Missing/extra matched train/eval seed pairs')
    seed_rows, by_condition = [], []
    for ts in manifest['train_seeds']:
        rows = [paired[(ts, es)] for es in manifest['eval_seeds']]
        result = {'train_seed': ts, 'n_eval_seeds': len(rows)}
        for metric in METRICS:
            cvals = [x['C'][metric] for x in rows]
            dvals = [x['D'][metric] for x in rows]
            diff = [b - a for a, b in zip(cvals, dvals)]
            result['C_' + metric] = statistics.mean(cvals)
            result['D_' + metric] = statistics.mean(dvals)
            result['D_minus_C_' + metric] = statistics.mean(diff)
            result['D_win_eval_count_' + metric] = sum(d > 0 for d in diff)
            result['D_tie_eval_count_' + metric] = sum(d == 0 for d in diff)
        seed_rows.append(result)
    for metric in METRICS:
        means_c = [r['C_' + metric] for r in seed_rows]
        means_d = [r['D_' + metric] for r in seed_rows]
        deltas = [r['D_minus_C_' + metric] for r in seed_rows]
        by_condition.append({'metric': metric, 'C_mean': statistics.mean(means_c),
            'D_mean': statistics.mean(means_d), 'D_minus_C': statistics.mean(deltas),
            'paired_delta_SD_between_training_seeds': stdev(deltas),
            'C_SD_between_training_seeds': stdev(means_c),
            'D_SD_between_training_seeds': stdev(means_d),
            'D_wins_training_seeds': sum(d > 0 for d in deltas),
            'ties_training_seeds': sum(d == 0 for d in deltas),
            'n_independent_training_seeds': len(deltas),
            'n_eval_seeds_per_train': len(manifest['eval_seeds'])})
    return seed_rows, by_condition


def write_csv(file, rows):
    with file.open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_report(folder: Path):
    manifest, rows = collect(folder)
    per_train, summary = summarize(manifest, rows)
    write_csv(folder / 'V6_all_evaluations.csv', rows)
    write_csv(folder / 'V6_per_training_seed.csv', per_train)
    write_csv(folder / 'V6_C_vs_D_summary.csv', summary)
    report = [
        '# B3-V6 — So sánh C và D: Edge-GAT + MAPPO', '',
        f"Training seeds (độc lập): {manifest['train_seeds']}",
        f"Evaluation seeds (held-out): {manifest['eval_seeds']}",
        f"Train: {manifest['train_episodes']} episodes × {manifest['train_sim_seconds']} s; "
        f"Eval: {manifest['eval_sim_seconds']} s/seed.", '',
        '| Chỉ số | C: Edge-GAT, prior 1.5 | D: Edge-GAT, prior 0.5 | D − C | SD(D − C) theo train seed |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in summary:
        std = r['paired_delta_SD_between_training_seeds']
        report.append(f"| {r['metric']} | {r['C_mean']:.3f} | {r['D_mean']:.3f} | "
                      f"{r['D_minus_C']:+.3f} | {std:.3f} |" if std is not None else
                      f"| {r['metric']} | {r['C_mean']:.3f} | {r['D_mean']:.3f} | {r['D_minus_C']:+.3f} | N/A |")
    report.extend(['', '## Theo từng training seed', '',
                   '| Train seed | Job C | Job D | D − C | Block C | Block D | D − C |',
                   '|---|---:|---:|---:|---:|---:|---:|'])
    for r in per_train:
        report.append(f"| {r['train_seed']} | {r['C_jobDone']:.3f} | {r['D_jobDone']:.3f} | "
                      f"{r['D_minus_C_jobDone']:+.3f} | {r['C_block']:.3f} | "
                      f"{r['D_block']:.3f} | {r['D_minus_C_block']:+.3f} |")
    report.extend(['', '## Quy tắc diễn giải', '',
       '- JobDone/TargetReached/reward: cao hơn là tốt; block/livelock/wrongTarget: thấp hơn là tốt.',
       '- Tính cặp D−C theo cùng training seed và evaluation seed; SD chỉ dựa trên **training seeds độc lập**.',
       '- Evaluation seeds chỉ là lặp điều kiện môi trường trên cùng checkpoint, **không phải checkpoint độc lập**.',
       '- Các giá trị trên mô tả hiệu suất. 3 training seeds quá ít để kết luận ý nghĩa thống kê một cách chắc chắn.',
       '- Nếu dùng seed 2001–2005 để chọn C hoặc D thì đây không còn là test set hoàn toàn chưa được dùng để chọn mô hình; khi cần báo cáo cuối hãy dùng seed mới.',
       '- Bản V6 train từ đầu; không load thêm 30 episode vào checkpoint V5 trước đó.',
       '- Kết quả trước/sau ở số episode khác nhau không cô lập được nguyên nhân tăng điểm là do train lâu hay kiến trúc.',
       '- Giữ Dispatching B2 MLP và MAPPO/CTDE; GNN chỉ dùng cho Routing.', ''])
    output = folder / 'V6_CD_REPORT.md'
    output.write_text('\n'.join(report), encoding='utf-8')
    print('\n'.join(report[:report.index('## Quy tắc diễn giải')]))
    print('\nSaved report:', output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, default=Path('mappo_runs/B3_V6_CD_30ep_720s'))
    a = p.parse_args()
    make_report(a.folder.resolve())


if __name__ == '__main__':
    main()
