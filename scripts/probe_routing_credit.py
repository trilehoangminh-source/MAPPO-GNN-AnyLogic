"""Inspect per-action reward/GAE of a fixed MAPPO checkpoint on Tobias Matrix.

No PPO updates, no checkpoint writes, no changes to other source files.
Important: collector.episode(train=True) SAMPLES actions but does not train;
only an explicit model.train(trajectories) would update neural weights.
These are observational aggregates, not causal action comparisons.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import train_mappo as trainer
from thesis.mappo import EventCollector


def get_args():
    p = argparse.ArgumentParser(description="Read routing rewards/advantages without PPO training")
    p.add_argument('--model', default='envs/Matrix_python39.zip')
    p.add_argument('--checkpoint', default='mappo_runs/check_720/latest_mappo.pt')
    p.add_argument('--output', default='mappo_runs/credit_debug')
    p.add_argument('--port', type=int, default=51320)
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--sim-seconds', type=int, default=120)
    p.add_argument('--max-events', type=int, default=20000)
    p.add_argument('--fleetsize', type=int, default=8)
    p.add_argument('--max-fleetsize', type=int, default=20)
    p.add_argument('--seed', type=int, default=45)
    p.add_argument('--device', default='cpu')
    p.add_argument('--env-type', default='matrix')
    p.add_argument('--routing-backbone', default='original_gnn')
    p.add_argument('--dispatch-backbone', default='original_linear')
    p.add_argument('--single-dispatch-policy', action='store_true')
    p.add_argument('--gamma', type=float, default=0.90)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--mini-batches', type=int, default=1)
    p.add_argument('--hidden', type=int, default=32)
    a = p.parse_args()
    if not Path(a.model).is_file():
        p.error(f'Model ZIP not found: {a.model}')
    if not Path(a.checkpoint).is_file():
        p.error(f'Checkpoint not found: {a.checkpoint}')
    return a


def stats(rows):
    if not rows:
        return {'count': 0}
    def metrics(field):
        arr = np.asarray([r[field] for r in rows], dtype=np.float64)
        return {'mean': round(float(arr.mean()), 5),
                'median': round(float(np.median(arr)), 5),
                'min': round(float(arr.min()), 5),
                'max': round(float(arr.max()), 5)}
    return {'count': len(rows),
            'reward': metrics('reward'),
            'raw_gae': metrics('gae'),
            'positive_gae_fraction': round(float(np.mean([r['gae'] > 0 for r in rows])), 4),
            'p_action0_at_decision': metrics('p0'),
            'mean_elapsed_seconds': round(float(np.mean([r['elapsed'] for r in rows])), 4)}


def main():
    a = get_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        print('[CREDIT] Starting model...', flush=True)
        env = trainer.build_env(a)
        model = trainer.make_model(env, a)
        model.load(str(Path(a.checkpoint).resolve()))
        print('[CREDIT] Checkpoint loaded; sampling actions, NO PPO updates.', flush=True)
        collector = EventCollector(model, env)
        trajectories, meta = collector.episode(seed=a.seed, train=True,
                                                max_event_steps=a.max_events)

        # Reproduce the exact GAE recursion from MAPPO.train() on closed transitions.
        from thesis.mappo.algorithm import as_actor, batchify
        import torch
        records = []
        gamma, lam = model.cfg.gamma, model.cfg.gae_lambda
        routing_actor = model.roles['routing'].actor
        routing_actor.eval()
        for trajectory in trajectories:
            gae = 0.0
            for step in reversed(trajectory):
                next_v = 0.0 if step['done'] else float(step['next_value'])
                delta = float(step['reward']) + gamma * next_v - float(step['value'])
                gae = delta + gamma * lam * (0.0 if step['done'] else gae)
                if step['role'] != 'routing':
                    continue
                with torch.no_grad():
                    obs = batchify(as_actor(step['observation'], model.device))
                    p0 = float(routing_actor.dist(obs).probs[0, 0].item())
                records.append({
                    'action': int(step['action']),
                    'reward': float(step['reward']),
                    'gae': float(gae),
                    'p0': p0,
                    'elapsed': float(step.get('elapsed_seconds', 0.0))
                })

        by_action = {str(i): stats([r for r in records if r['action'] == i])
                     for i in range(5)}
        stopped = stats([r for r in records if r['action'] == 0])
        moved = stats([r for r in records if r['action'] != 0])
        report = {
            'notes': [
                'SAMPLED-policy evaluation, not greedy, and no model.train() call.',
                'Only CLOSED transitions with observed rewards are counted.',
                'Action-group rewards and GAE are observational; state/context differ between groups.',
                'GAE is from the LOADED checkpoint, not the original training rollout.',
            ],
            'events': meta.get('events'),
            'closed_transitions': meta.get('closed_transitions'),
            'discarded_pending': meta.get('discarded_pending'),
            'targetReached': (meta.get('statistics') or {}).get('targetReached'),
            'jobDone': (meta.get('statistics') or {}).get('jobDone'),
            'routing_samples': len(records),
            'by_action': by_action,
            'standstill': stopped,
            'movement': moved,
        }
        path = output / 'routing_credit.json'
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        print('\n========== ROUTING ACTION CREDIT ==========')
        print(f"Events={report['events']} | closed transitions={report['closed_transitions']} | pending={report['discarded_pending']}")
        print(f"targetReached={report['targetReached']} | jobDone={report['jobDone']}")
        print('Action  N    Mean reward   Mean raw GAE    Positive GAE')
        for k, v in by_action.items():
            if not v['count']:
                print(f'  {k}      0       n/a            n/a            n/a')
            else:
                print(f"  {k}    {v['count']:4d}   {v['reward']['mean']:10.4f}   {v['raw_gae']['mean']:12.4f}   {v['positive_gae_fraction']*100:8.1f}%")
        print(f"[CREDIT] Saved: {path}")
        print('NOTE: Per-action averages do NOT prove what reward an alternative action would have obtained.')
        print('===========================================\n')
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
