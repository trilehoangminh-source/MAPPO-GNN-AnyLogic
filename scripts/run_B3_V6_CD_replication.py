"""Train and evaluate C/D Edge-GAT at constant Dijkstra strengths, independent seeds.

Training ALWAYS starts fresh for each (condition, seed). Existing completed runs are
skipped only with --resume; incomplete runs are NEVER overwritten automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from v6_cd_common import (CONDITIONS, condition_dir, evaluation_cmd,
                          seed_validation, training_cmd)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='envs/Matrix_critic_fresh.zip')
    p.add_argument('--output', default='mappo_runs/B3_V6_CD_30ep_720s')
    p.add_argument('--train-seeds', type=int, nargs='+', default=[43, 143, 243])
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[2001, 2002, 2003, 2004, 2005])
    p.add_argument('--train-episodes', type=int, default=30)
    p.add_argument('--train-sim-seconds', type=int, default=720)
    p.add_argument('--eval-sim-seconds', type=int, default=3600)
    p.add_argument('--max-events', type=int, default=20000)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--graph-layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--port-train', type=int, default=51830)
    p.add_argument('--port-eval', type=int, default=51831)
    p.add_argument('--mode', choices=['all', 'train', 'eval'], default='all')
    p.add_argument('--resume', action='store_true', help='Skip only validated completed train/eval work')
    p.add_argument('--dry-run', action='store_true', help='Preview all commands without writing files')
    a = p.parse_args(argv)
    seed_validation(a.train_seeds, a.eval_seeds, a.train_episodes)
    if min(a.train_episodes, a.train_sim_seconds, a.eval_sim_seconds, a.max_events,
           a.hidden, a.graph_layers, a.heads, a.epochs) < 1 or a.hidden % a.heads:
        p.error('Episode/timeout/network arguments must be positive; hidden multiple of heads')
    if a.port_train == a.port_eval or any(not 1 <= port <= 65535 for port in (a.port_train, a.port_eval)):
        p.error('Training/evaluation ports must be different, within 1..65535')
    return a


def manifest(a, model):
    return {
        'protocol': 'V6_CD_EdgeGAT_constant_guide_v1',
        'train_seeds': a.train_seeds, 'eval_seeds': a.eval_seeds,
        'train_episodes': a.train_episodes, 'train_sim_seconds': a.train_sim_seconds,
        'eval_sim_seconds': a.eval_sim_seconds, 'max_events': a.max_events,
        'hidden': a.hidden, 'graph_layers': a.graph_layers, 'heads': a.heads,
        'epochs': a.epochs, 'model_file': str(model),
        'conditions': [{'code': x.code, 'encoder': 'residual_edge_gat',
                        'guide': x.guide} for x in CONDITIONS],
        'protocol_note': 'Independent train per seed; constant guide; held-out eval; B2 Operation-first MLP dispatch; MAPPO CTDE',
        # Deliberately exclude mode, output, ports and resume from reproducibility definition.
    }


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def valid_train(folder: Path, condition, ts: int, a, model: Path) -> bool:
    ckpt = folder / f'checkpoint_ep_{a.train_episodes:04d}.pt'
    config = folder / 'run_config.json'
    log = folder / 'metrics.jsonl'
    if not (ckpt.is_file() and ckpt.stat().st_size > 1024 and config.is_file() and log.is_file()):
        return False
    c = json.loads(config.read_text(encoding='utf-8'))
    required = {'version': 'B3_EDGE_GAT_MAPPO_V4', 'routing_encoder': 'residual_edge_gat',
                'dispatch_backbone': 'pooled', 'dispatch_mask_mode': 'operation_first',
                'seed': ts, 'episodes': a.train_episodes, 'sim_seconds': a.train_sim_seconds,
                'max_events': a.max_events, 'hidden': a.hidden,
                'graph_layers': a.graph_layers, 'heads': a.heads, 'epochs': a.epochs,
                'mini_batches': 1, 'stop_mask_mode': 'forbid_when_move'}
    for key, expected in required.items():
        if c.get(key) != expected:
            raise RuntimeError(f'Training config mismatch {key}: {config}; got {c.get(key)!r}, want {expected!r}')
    for key, expected in [('guide_start', condition.guide), ('guide_final', condition.guide),
                          ('final_used_guide_strength', condition.guide),
                          ('gamma', .9), ('gae_lambda', .95), ('clip', .1), ('lr', .0001)]:
        if not math.isclose(float(c.get(key, float('nan'))), expected, abs_tol=1e-10):
            raise RuntimeError(f'Training config mismatch {key}: {config}')
    if Path(c['model_file']).resolve() != model.resolve():
        raise RuntimeError('AnyLogic model path mismatch: ' + str(config))
    rows = read_jsonl(log)
    if len(rows) != a.train_episodes or [r.get('episode') for r in rows] != list(range(1, a.train_episodes + 1)):
        return False
    if [r.get('environment_seed') for r in rows] != [ts + i for i in range(a.train_episodes)]:
        return False
    return True


def valid_eval(path: Path, condition, a, checkpoint: Path) -> bool:
    if not path.is_file():
        return False
    rows = read_jsonl(path)
    if len(rows) != len(a.eval_seeds) or [r.get('seed') for r in rows] != a.eval_seeds:
        return False
    for r in rows:
        e = r.get('environment', {})
        if (r.get('evaluation') is not True or r.get('version') != 'B3_EDGE_GAT_MAPPO_V4'
            or not e.get('reached_terminal', False) or
            not math.isclose(float(e.get('dijkstra_strength_eval', -1)), condition.guide, abs_tol=1e-10)
            or Path(r.get('checkpoint', '')).resolve() != checkpoint.resolve()):
            return False
    return True


def logged_call(cmd, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('\n>>>', subprocess.list2cmdline(cmd), flush=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with log.open('x', encoding='utf-8') as f:
        f.write('COMMAND: ' + subprocess.list2cmdline(cmd) + '\n\n')
        f.flush()
        process = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            f.write(line)
        code = process.wait()
        f.write(f'\nEXIT_CODE={code}\n')
    if code:
        raise RuntimeError(f'Subprocess failed ({code}), see {log}. Do not remove partial run automatically.')


def run(a):
    model = (ROOT / a.model).resolve()
    folder = (ROOT / a.output).resolve()
    for script in ('train_B3_edgeGAT_MAPPO_v4.py', 'evaluate_B3_edgeGAT_MAPPO_v4.py'):
        if not (ROOT / 'scripts' / script).is_file():
            raise FileNotFoundError(f'Missing V4 script: scripts/{script}')
    if not a.dry_run and not model.is_file():
        raise FileNotFoundError(f'AnyLogic model missing: {model}')
    info = manifest(a, model)
    mf = folder / 'experiment_manifest.json'
    if mf.exists() and json.loads(mf.read_text(encoding='utf-8')) != info:
        raise RuntimeError('Output folder contains a DIFFERENT protocol: ' + str(mf))
    if not a.dry_run:
        folder.mkdir(parents=True, exist_ok=True)
        if not mf.exists():
            mf.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    for condition in CONDITIONS:
        for ts in a.train_seeds:
            run_folder = condition_dir(folder, condition, ts)
            ckpt = run_folder / f'checkpoint_ep_{a.train_episodes:04d}.pt'
            eval_file = run_folder / f'evaluation_{a.eval_sim_seconds}.jsonl'
            tcmd = training_cmd(sys.executable, ROOT, model, condition, ts, a.train_episodes,
                                a.train_sim_seconds, a.max_events, a.port_train, a.hidden,
                                a.graph_layers, a.heads, a.epochs, run_folder)
            ecmd = evaluation_cmd(sys.executable, ROOT, ckpt, a.eval_seeds, a.eval_sim_seconds,
                                  a.port_eval, eval_file)
            if a.dry_run:
                print(f'\n{condition.code} train_seed={ts} guide={condition.guide}')
                print('TRAIN:', subprocess.list2cmdline(tcmd))
                print('EVAL: ', subprocess.list2cmdline(ecmd))
                continue
            if a.mode != 'eval':
                if run_folder.exists() and any(run_folder.iterdir()):
                    if not a.resume or not valid_train(run_folder, condition, ts, a, model):
                        raise RuntimeError(f'Partial/existing training folder, refusing overwrite: {run_folder}')
                    print('[SKIP VERIFIED TRAIN]', run_folder, flush=True)
                else:
                    logged_call(tcmd, folder / 'logs' / f'{condition.code}_train_seed{ts}.log')
                    if not valid_train(run_folder, condition, ts, a, model):
                        raise RuntimeError(f'Incomplete training data: {run_folder}')
            else:
                if not valid_train(run_folder, condition, ts, a, model):
                    raise RuntimeError('Cannot evaluate missing/incompatible checkpoint: ' + str(run_folder))
            if a.mode != 'train':
                if eval_file.exists():
                    if not a.resume or not valid_eval(eval_file, condition, a, ckpt):
                        raise RuntimeError('Partial/existing evaluation, refusing overwrite: ' + str(eval_file))
                    print('[SKIP VERIFIED EVAL]', eval_file, flush=True)
                else:
                    logged_call(ecmd, folder / 'logs' / f'{condition.code}_eval_seed{ts}.log')
                    if not valid_eval(eval_file, condition, a, ckpt):
                        raise RuntimeError('Incomplete evaluation file: ' + str(eval_file))
    if a.dry_run:
        print('\nDRY-RUN: No files changed. Train 2 configurations x', len(a.train_seeds),
              'seeds x', a.train_episodes, 'episodes; evaluate',
              2 * len(a.train_seeds) * len(a.eval_seeds), 'episodes.')
    else:
        print('\nCOMPLETE:', a.mode, '\nOutput:', folder)
        if a.mode != 'train':
            print('Next: python scripts/analyze_B3_V6_CD_replication.py --folder', folder)


if __name__ == '__main__':
    run(arguments())
