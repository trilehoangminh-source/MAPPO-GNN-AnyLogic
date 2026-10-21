"""Four-arm 2x2 factorial ablation on trained Routing policies (no algorithm edits).

A GAT+1.5; B GAT+0.5; C Edge-GAT+1.5; D Edge-GAT+0.5.
Same B2 dispatch, MAPPO, train seeds, evaluation seeds, sim time, optimizer settings.
Separate TRAINING for each arm; no inference-only change of Dijkstra strength.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ablation_abcd_v5_common import (ARMS, assert_distinct_seeds, run_dir,
                                   train_command, eval_command)


def settings(a, model: Path):
    return {'protocol': 'B3_ABCD_V5_factorial_v1', 'train_seeds': a.train_seeds,
            'eval_seeds': a.eval_seeds, 'train_episodes': a.train_episodes,
            'train_sim_seconds': a.train_sim_seconds, 'eval_sim_seconds': a.eval_sim_seconds,
            'max_events': a.max_events, 'hidden': a.hidden,
            'graph_layers': a.graph_layers, 'heads': a.heads, 'epochs': a.epochs,
            'model_file': str(model),
            'factors': [{'arm': x.code, 'encoder': x.encoder,
                         'dijkstra_logit_strength': x.guide} for x in ARMS],
            'architecture': 'Routing only: GAT/EdgeGAT; Dispatching B2 pooled/MLP',
            'teacher_schedule': 'constant throughout training and evaluation',
            'reuse': 'fresh training; compare after held-out evaluation; no checkpoint overwrite'}


def check_completed_train(directory: Path, arm, seed: int, a) -> bool:
    checkpoint = directory / f'checkpoint_ep_{a.train_episodes:04d}.pt'
    configfile = directory / 'run_config.json'
    metrics = directory / 'metrics.jsonl'
    if not (checkpoint.is_file() and configfile.is_file() and metrics.is_file()):
        return False
    c = json.loads(configfile.read_text(encoding='utf-8'))
    if (int(c['seed']) != seed or int(c['episodes']) != a.train_episodes
            or int(c['sim_seconds']) != a.train_sim_seconds
            or int(c['graph_layers']) != a.graph_layers
            or int(c['hidden']) != a.hidden
            or int(c['max_events']) != a.max_events
            or c.get('dispatch_backbone') != 'pooled'
            or c.get('dispatch_mask_mode') != 'operation_first'):
        raise RuntimeError(f'Incompatible checkpoint config: {configfile}')
    if arm.encoder == 'gat':
        if (c.get('routing_encoder') != 'gat'
                or abs(float(c['routing_guide_strength']) - arm.guide) > 1e-9
                or int(c['gat_heads']) != a.heads):
            raise RuntimeError(f'Incompatible GAT arm: {directory}')
    else:
        if (c.get('routing_encoder') != 'residual_edge_gat'
                or abs(float(c['guide_start']) - arm.guide) > 1e-9
                or abs(float(c['guide_final']) - arm.guide) > 1e-9
                or int(c['heads']) != a.heads):
            raise RuntimeError(f'Incompatible Edge-GAT arm: {directory}')
    rows = [json.loads(s) for s in metrics.read_text(encoding='utf-8').splitlines() if s.strip()]
    return len(rows) == a.train_episodes and rows[-1].get('episode') == a.train_episodes


def check_completed_eval(out: Path, a) -> bool:
    if not out.is_file():
        return False
    rows = [json.loads(s) for s in out.read_text(encoding='utf-8').splitlines() if s.strip()]
    return len(rows) == len(a.eval_seeds) and [int(r['seed']) for r in rows] == a.eval_seeds


def run_logged(cmd: list[str], log: Path, *, cwd: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('\n>>', subprocess.list2cmdline(cmd), flush=True)
    with log.open('w', encoding='utf-8') as f:
        f.write('Command: ' + subprocess.list2cmdline(cmd) + '\n\n')
        f.flush()
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding='utf-8', errors='replace', bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end='', flush=True)
            f.write(line)
        rc = p.wait()
        f.write(f'\nEXIT_CODE={rc}\n')
    if rc != 0:
        raise RuntimeError(f'Command failed with exit code {rc}. See log: {log}')


def run_experiment(a) -> None:
    model = (ROOT / a.model).resolve()
    outroot = (ROOT / a.output).resolve()
    assert_distinct_seeds(a.train_seeds, a.eval_seeds, a.train_episodes)
    if not a.dry_run and not model.is_file():
        raise FileNotFoundError(f'AnyLogic ZIP not found: {model}')
    for arm in ARMS:
        for script in [arm.train_script, arm.eval_script]:
            if not (ROOT / 'scripts' / script).is_file():
                raise FileNotFoundError(f'Missing base V2/V4 script: {ROOT / "scripts" / script}')
    manifest = settings(a, model)
    manifestfile = outroot / 'experiment_manifest.json'
    if manifestfile.exists():
        old = json.loads(manifestfile.read_text(encoding='utf-8'))
        if old != manifest:
            raise RuntimeError(f'Output belongs to DIFFERENT experiment settings: {manifestfile}. '
                               'Choose a new --output folder.')
    if not a.dry_run:
        outroot.mkdir(parents=True, exist_ok=True)
        manifestfile.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for arm in ARMS:
        for seed in a.train_seeds:
            folder = run_dir(outroot, arm, seed)
            checkpoint = folder / f'checkpoint_ep_{a.train_episodes:04d}.pt'
            evout = folder / f'evaluation_{a.eval_sim_seconds}.jsonl'
            train_cmd = train_command(sys.executable, ROOT, model, arm, seed,
                                      a.train_episodes, a.train_sim_seconds,
                                      a.max_events, a.port_train, a.hidden,
                                      a.graph_layers, a.heads, folder, epochs=a.epochs)
            eval_cmd = eval_command(sys.executable, ROOT, arm, checkpoint,
                                    a.eval_seeds, a.eval_sim_seconds, a.port_eval, evout)
            if a.dry_run:
                print('\n###', arm.code, arm.encoder, 'guide', arm.guide, 'train_seed', seed)
                print('TRAIN:', subprocess.list2cmdline(train_cmd))
                print('EVAL :', subprocess.list2cmdline(eval_cmd))
                continue
            if a.mode != 'eval':
                if folder.exists() and any(folder.iterdir()):
                    if not (a.resume and check_completed_train(folder, arm, seed, a)):
                        raise RuntimeError(f'Existing/partial train folder, REFUSE overwrite: {folder}')
                    print('[SKIP completed train]', folder, flush=True)
                else:
                    run_logged(train_cmd, outroot / 'logs' / f'{arm.code}_seed{seed}_train.log', cwd=ROOT)
                    if not check_completed_train(folder, arm, seed, a):
                        raise RuntimeError(f'Incomplete training after success exit: {folder}')
            else:
                if not check_completed_train(folder, arm, seed, a):
                    raise FileNotFoundError(f'Cannot evaluate incomplete training: {folder}')
            if a.mode != 'train':
                if evout.exists():
                    if not (a.resume and check_completed_eval(evout, a)):
                        raise RuntimeError(f'Existing/partial evaluation, REFUSE overwrite: {evout}')
                    print('[SKIP completed eval]', evout, flush=True)
                else:
                    run_logged(eval_cmd, outroot / 'logs' / f'{arm.code}_seed{seed}_eval.log', cwd=ROOT)
                    if not check_completed_eval(evout, a):
                        raise RuntimeError(f'Incomplete evaluation after success exit: {evout}')
    if a.dry_run:
        print('\nDRY RUN ONLY: no training or evaluation executed.')
    else:
        print('\nCOMPLETE requested stage. Results in:', outroot)
        if a.mode != 'train':
            print('Next: python scripts/analyze_B3_ablation_ABCD_v5.py --folder', outroot)


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='mappo_runs/B3_ABCD_V5_pilot',
                   help='New experiment folder (must not contain incompatible results)')
    p.add_argument('--model', default='envs/Matrix_critic_fresh.zip')
    p.add_argument('--train-seeds', type=int, nargs='+', default=[43])
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[1001, 1002, 1003])
    p.add_argument('--train-episodes', type=int, default=2)
    p.add_argument('--train-sim-seconds', type=int, default=720)
    p.add_argument('--eval-sim-seconds', type=int, default=3600)
    p.add_argument('--max-events', type=int, default=20000)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--graph-layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--port-train', type=int, default=51780)
    p.add_argument('--port-eval', type=int, default=51781)
    p.add_argument('--mode', choices=['all', 'train', 'eval'], default='all')
    p.add_argument('--resume', action='store_true', help='Skip only COMPLETE and VERIFIED runs')
    p.add_argument('--dry-run', action='store_true', help='Print exact commands; do not run')
    a = p.parse_args()
    if (a.hidden < 4 or a.heads < 1 or a.hidden % a.heads or
        min(a.graph_layers, a.epochs, a.train_episodes, a.train_sim_seconds,
            a.eval_sim_seconds, a.max_events) < 1):
        p.error('Invalid network size, episode count or training/evaluation budget')
    if a.port_train == a.port_eval:
        p.error('Train and eval ports must differ')
    return a


if __name__ == '__main__':
    run_experiment(get_args())
