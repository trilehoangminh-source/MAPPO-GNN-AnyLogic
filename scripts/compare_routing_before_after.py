"""Compare untrained vs checkpointed Routing Actor ON THE SAME observations.

Run from your Tobias MAPPO thesis project root (Python .venv39, Java 11):

  python -u scripts/compare_routing_before_after.py \
      --model envs/Matrix_python39.zip \
      --checkpoint mappo_runs/check_720/latest_mappo.pt \
      --sim-seconds 120 --port 51310 \
      --output mappo_runs/compare_routing

Creates the original, seeded untrained Actor, makes a copy, and then loads the
checkpoint into the actual MAPPO model. During ONE greedy evaluation of the
trained policy, evaluates BOTH Actors on exactly the same observations.

This is a diagnostic: never calls model.train() or saves a checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_mappo as trainer
from thesis.mappo import EventCollector
from thesis.mappo.algorithm import as_actor, batchify


def get_args():
    p = argparse.ArgumentParser(description="Compare initial vs trained Routing Actor on identical observations")
    p.add_argument('--model', default='envs/Matrix_python39.zip')
    p.add_argument('--checkpoint', default='mappo_runs/check_720/latest_mappo.pt')
    p.add_argument('--output', default='mappo_runs/compare_routing')
    p.add_argument('--port', type=int, default=51310)
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--sim-seconds', type=int, default=120)
    p.add_argument('--max-events', type=int, default=20000)
    p.add_argument('--fleetsize', type=int, default=8)
    p.add_argument('--max-fleetsize', type=int, default=20)
    p.add_argument('--seed', type=int, default=45, help='AnyLogic simulation seed')
    p.add_argument('--init-seed', type=int, default=43, help='Original training network-initialization seed')
    p.add_argument('--device', default='cpu')
    p.add_argument('--env-type', default='matrix', choices=['matrix', 'minimatrix'])
    p.add_argument('--routing-backbone', choices=['original_gnn', 'pooled'], default='original_gnn')
    p.add_argument('--dispatch-backbone', choices=['original_linear', 'pooled'], default='original_linear')
    p.add_argument('--single-dispatch-policy', action='store_true')
    p.add_argument('--gamma', type=float, default=0.90)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--mini-batches', type=int, default=1)
    p.add_argument('--hidden', type=int, default=32)
    args = p.parse_args()
    if not Path(args.model).is_file():
        p.error('Model ZIP not found: '+args.model)
    if not Path(args.checkpoint).is_file():
        p.error('MAPPO checkpoint not found: '+args.checkpoint)
    return args


def describe(rows, column):
    probs = np.stack([r[column] for r in rows])
    p0 = probs[:, 0]
    choices = probs.argmax(axis=1)
    n = len(rows)
    counts = {str(a): int(np.sum(choices == a)) for a in range(5)}
    return {
        'mean_probability_action_0': round(float(p0.mean()), 6),
        'median_probability_action_0': round(float(np.median(p0)), 6),
        'greedy_action_counts': counts,
        'greedy_stop_percent': round(100.0 * counts['0'] / n, 2),
        'p_action0_over_50pct': int(np.sum(p0 >= 0.5)),
        'mean_probabilities_all_actions': [round(float(x), 5) for x in probs.mean(axis=0)],
    }


def main():
    args = get_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Match the initialization seed used by train_mappo.py (default 43).
    random.seed(args.init_seed)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)

    env = None
    try:
        print('[COMPARE] Starting AnyLogic...', flush=True)
        env = trainer.build_env(args)
        print('[COMPARE] Creating UNTRAINED MAPPO Actor with seed', args.init_seed, flush=True)
        model = trainer.make_model(env, args)
        # Snapshot weights BEFORE loading the checkpoint.
        untrained_actor = copy.deepcopy(model.roles['routing'].actor)
        untrained_actor.eval()

        print('[COMPARE] Loading TRAINED checkpoint...', flush=True)
        model.load(str(Path(args.checkpoint).resolve()))
        trained_actor = model.roles['routing'].actor
        trained_actor.eval()

        with torch.no_grad():
            squared = sum(
                float((p_init - p_trained).square().sum().item())
                for p_init, p_trained in zip(untrained_actor.parameters(), trained_actor.parameters())
            )
            weight_delta_l2 = squared ** 0.5
        print(f'[COMPARE] Actor weight difference (L2) = {weight_delta_l2:.6f}', flush=True)

        # The evaluation rollout follows ONLY the trained Actor.
        # The untrained Actor receives the EXACT SAME OBSERVATIONS, but
        # its hypothetical actions are NOT submitted to AnyLogic.
        saved_act = model.act
        rows = []

        def compare_act(role, obs, state, greedy=False):
            if role == 'routing':
                actor_obs = batchify(as_actor(obs, model.device))
                with torch.no_grad():
                    untrained_actor.eval()
                    trained_actor.eval()
                    p_init = untrained_actor.dist(actor_obs).probs[0].detach().cpu().numpy()
                    p_trained = trained_actor.dist(actor_obs).probs[0].detach().cpu().numpy()
                mask = np.asarray(obs['action_mask']).reshape(-1) > 0.5
                if p_init.shape != (5,) or p_trained.shape != (5,) or mask.shape != (5,):
                    raise ValueError('Expected 5 Routing actions, got '+str((p_init.shape, p_trained.shape, mask.shape)))
                rows.append({
                    'initial': p_init.copy(),
                    'trained': p_trained.copy(),
                    'legal_move_exists': bool(mask[1:].any()),
                })
            return saved_act(role, obs, state, greedy=greedy)

        model.act = compare_act
        print('[COMPARE] Evaluating ONE episode (greedy trained policy; NO training)...', flush=True)
        collector = EventCollector(model, env)
        _, meta = collector.episode(seed=args.seed, train=False, max_event_steps=args.max_events)
        if not rows:
            print('[COMPARE] No Routing observations collected.', flush=True)
            return

        initial = describe(rows, 'initial')
        trained = describe(rows, 'trained')
        initial_choices = np.array([r['initial'].argmax() for r in rows])
        trained_choices = np.array([r['trained'].argmax() for r in rows])
        action_changed = int(np.sum(initial_choices != trained_choices))
        n = len(rows)
        legal_moves = sum(int(r['legal_move_exists']) for r in rows)

        summary = {
            'description': 'Both Actors evaluated on identical observations from trained-policy greedy rollout',
            'caution': 'Initial Actor hypothetical choices were not executed; these are not two separate production-performance evaluations.',
            'initial_network_seed': args.init_seed,
            'simulation_seed': args.seed,
            'simulation_seconds': args.sim_seconds,
            'checkpoint': str(args.checkpoint),
            'routing_observations': n,
            'legal_movement_observations': legal_moves,
            'actor_weight_delta_l2': weight_delta_l2,
            'hypothetical_argmax_action_changed': action_changed,
            'untrained_actor': initial,
            'trained_actor': trained,
            'evaluation_statistics': {
                'events': meta.get('events'),
                'targetReached': (meta.get('statistics') or {}).get('targetReached'),
                'jobDone': (meta.get('statistics') or {}).get('jobDone'),
                'mask_violations': (meta.get('action_diagnostics') or {}).get('selected_mask_violations'),
            },
        }
        target = output/'routing_init_vs_trained.json'
        target.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

        print('\n=========== ROUTING INIT vs TRAINED ===========')
        print('[COMPARE] Observations (SAME inputs):', n)
        print('[COMPARE] Allowed movement on:', f'{legal_moves}/{n}')
        print('[COMPARE] UNTRAINED: mean P(a0) =', f"{initial['mean_probability_action_0']:.4f}",
              '| greedy a0 =', f"{initial['greedy_stop_percent']:.1f}%")
        print('[COMPARE] UNTRAINED argmax counts:', initial['greedy_action_counts'])
        print('[COMPARE] TRAINED  : mean P(a0) =', f"{trained['mean_probability_action_0']:.4f}",
              '| greedy a0 =', f"{trained['greedy_stop_percent']:.1f}%")
        print('[COMPARE] TRAINED argmax counts:', trained['greedy_action_counts'])
        print('[COMPARE] Argmax changed on:', f'{action_changed}/{n}', 'observations')
        print('[COMPARE] Actor weights changed (L2):', f'{weight_delta_l2:.6f}')
        print('[COMPARE] AnyLogic targetReached:', summary['evaluation_statistics']['targetReached'],
              '| JobDone:', summary['evaluation_statistics']['jobDone'])
        print('[COMPARE] Saved:', target)
        print('NOTE: Untrained counts are hypothetical actions on trained-policy observations.')
        print('================================================\n')
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
