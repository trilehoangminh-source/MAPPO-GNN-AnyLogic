"""Inspect Routing policy probabilities across saved MAPPO checkpoints.

Works with the exact pooled MAPPO A/Alpyne 1.2.1 project used in
train_mappo_long_v121.py and evaluate_mappo_long_v121.py.

Performs evaluation only (NO optimization, checkpoint writes, or changes to
original code). For each checkpoint, reloads weights and evaluates the same
held-out seed with the same simulation horizon. Reports P(Stop), argmax(Stop),
actual Stop, and environment outcomes. Sampling mode does not call model.train.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import train_mappo_stop_ab as trainer
from thesis.mappo.algorithm import as_actor, batchify
from thesis.mappo.collector_stop_ab import EventCollector


def summarize(probs, chosen, masks):
    if not probs:
        return {'routing_decisions': 0}
    arr = np.asarray(probs, dtype=float)
    mask = np.asarray(masks, dtype=bool)
    chosen = np.asarray(chosen, dtype=int)
    has_move = mask[:, 1:].any(axis=1)
    max_movement_prob = np.max(np.where(mask[:, 1:], arr[:, 1:], -1), axis=1)
    p0 = arr[:, 0]
    argmax = arr.argmax(axis=1)
    n = arr.shape[0]
    return {
        'routing_decisions': n,
        'move_available_decisions': int(has_move.sum()),
        'selected_stop_pct': round(100 * float(np.mean(chosen == 0)), 3),
        'greedy_stop_pct': round(100 * float(np.mean(argmax == 0)), 3),
        'mean_p_stop': round(float(p0.mean()), 6),
        'median_p_stop': round(float(np.median(p0)), 6),
        'min_p_stop': round(float(p0.min()), 6),
        'max_p_stop': round(float(p0.max()), 6),
        'p_stop_ge_0_5_pct': round(100 * float(np.mean(p0 >= 0.5)), 3),
        'mean_p_stop_minus_best_move_when_move_legal': (
            round(float(np.mean((p0 - max_movement_prob)[has_move])), 6) if has_move.any() else None
        ),
        'selected_action_counts': {str(i): int((chosen == i).sum()) for i in range(5)},
    }


def main():
    p = argparse.ArgumentParser(description='Inspect Routing action probabilities without training')
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--sim-seconds', type=int, default=120)
    p.add_argument('--port', type=int, default=51640)
    p.add_argument('--mode', choices=('greedy', 'sample'), default='greedy')
    p.add_argument('--output', type=Path, default=Path('mappo_runs/diagnose_stop_probabilities.json'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--lock-timeout', type=int, default=180)
    p.add_argument('--java-exe', default=r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
    opt = p.parse_args()
    paths = [x.resolve() for x in opt.checkpoints]
    if len(paths) != len(set(paths)):
        p.error('Do not repeat checkpoints.')
    if not all(x.is_file() for x in paths):
        p.error('A checkpoint file was not found: ' + ', '.join(str(x) for x in paths if not x.is_file()))
    configs = []
    for ckpt in paths:
        manifest = ckpt.parent / 'run_config.json'
        if not manifest.is_file():
            p.error(f'Missing manifest for {ckpt}: {manifest}')
        configs.append(json.loads(manifest.read_text(encoding='utf8')))
    c = configs[0]
    if any(x != c for x in configs):
        p.error('Checkpoint manifests differ; compare checkpoints from the same training configuration.')
    if (c['routing_backbone'], c['dispatch_backbone'], c['stop_mask_mode']) != ('pooled', 'pooled', 'baseline'):
        p.error('Only pooled + baseline A checkpoints are supported by this diagnostic.')
    if opt.seed in range(c['seed'], c['seed'] + c['episodes']):
        p.error('Use a held-out evaluation seed, not a training seed.')
    if opt.sim_seconds <= 0:
        p.error('--sim-seconds must be positive.')
    out = opt.output.resolve()
    if out.exists():
        p.error(f'Output already exists; choose another --output: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    if not Path(c['model_file']).is_file():
        p.error(f"AnyLogic model ZIP missing: {c['model_file']}")

    args = SimpleNamespace(
        model=c['model_file'], port=opt.port, startup_timeout=opt.startup_timeout,
        lock_timeout=opt.lock_timeout, java_exe=opt.java_exe,
        fleetsize=c['fleetsize'], max_fleetsize=c['max_fleetsize'],
        sim_seconds=opt.sim_seconds, max_events=c['max_events'],
        routing_backbone='pooled', dispatch_backbone='pooled',
        single_dispatch_policy=c['single_dispatch_policy'],
        gamma=c['gamma'], gae_lambda=c['gae_lambda'], clip=c['clip'],
        lr=c['lr'], epochs=c['epochs'], mini_batches=c['mini_batches'],
        hidden=c['hidden'], env_type='matrix', device=opt.device, seed=opt.seed,
    )
    env = None
    rows = []
    try:
        env = trainer.build_env(args)
        model = trainer.make_model(env, args)
        original_act = model.act
        samples = {'probs': [], 'chosen': [], 'masks': []}

        def instrumented_act(role, obs, state, greedy=False):
            if role != 'routing':
                return original_act(role, obs, state, greedy=greedy)
            actor = model.roles[role].actor
            actor.eval()
            with torch.no_grad():
                actor_input = batchify(as_actor(obs, model.device))
                probs = actor.dist(actor_input).probs[0].detach().cpu().numpy().copy()
            mask = np.asarray(obs['action_mask'], dtype=float).reshape(-1) > 0.5
            action, logp, value = original_act(role, obs, state, greedy=greedy)
            samples['probs'].append(probs)
            samples['masks'].append(mask)
            samples['chosen'].append(action)
            return action, logp, value

        model.act = instrumented_act
        for ckpt in paths:
            # Set the same Python/NumPy/Torch random state for comparable sampling.
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            torch.manual_seed(opt.seed)
            model.load(str(ckpt))
            for v in samples.values():
                v.clear()
            collector = EventCollector(model, env, stop_mask_mode='baseline')
            # train=True means SAMPLE actions here, not gradient updates.
            # There is deliberately no model.train(...) call in this script.
            _, meta = collector.episode(seed=opt.seed, train=(opt.mode == 'sample'), max_event_steps=args.max_events)
            d = summarize(samples['probs'], samples['chosen'], samples['masks'])
            stats = meta.get('statistics') or {}
            row = {
                'checkpoint': str(ckpt), 'seed': opt.seed, 'mode': opt.mode,
                'sim_seconds': opt.sim_seconds, 'policy': d,
                'environment': {k: stats.get(k) for k in ('jobDone','targetReached','wrongTarget','taken_invalid','newPart')},
                'delivered_reward': meta['delivered_reward'],
                'end_reason': meta['end_reason'],
                'actions_by_role': meta['action_diagnostics']['actions_by_role'],
            }
            rows.append(row)
            print(f"[STOP] {ckpt.name} mode={opt.mode} n={d.get('routing_decisions')} "
                  f"p0_mean={d.get('mean_p_stop')} p0_median={d.get('median_p_stop')} "
                  f"greedy_stop={d.get('greedy_stop_pct')}% selected_stop={d.get('selected_stop_pct')}% "
                  f"targetReached={stats.get('targetReached', 0)} jobDone={stats.get('jobDone', 0)}", flush=True)
    finally:
        if env is not None:
            env.close()
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
    print('[DONE] Saved ' + str(out), flush=True)


if __name__ == '__main__':
    main()
