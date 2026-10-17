"""Inspect actual Routing action probabilities from a saved MAPPO checkpoint.

Place beside scripts/probe_part_flow.py in the thesis project's scripts folder.
Run with the SAME arguments as probe_part_flow.py. No model training or
modification. Prints diagnostics at the end.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from thesis.mappo.algorithm import MAPPO, as_actor, batchify
import probe_part_flow

_original_act = MAPPO.act
_probability_rows = []


def observed_act(self, role, obs, state, greedy=False):
    # Read the same action distribution that RolePolicy.act uses.
    if role == 'routing':
        actor = self.roles[role].actor
        actor.eval()
        with torch.no_grad():
            ob = batchify(as_actor(obs, self.device))
            probs = actor.dist(ob).probs[0].detach().cpu().numpy()

        valid = np.asarray(obs['action_mask']).reshape(-1) > 0.5
        if len(probs) != 5 or len(valid) != 5:
            raise ValueError('Expected exactly five Routing actions (0..4)')
        legal_movement = bool(valid[1:].any())
        top = int(np.argmax(probs))
        legal_probs = np.sort(probs[valid])[::-1]
        runner_up = float(legal_probs[1]) if len(legal_probs) > 1 else 0.0
        _probability_rows.append({
            'p0': float(probs[0]),
            'pmax': float(probs[top]),
            'argmax': top,
            'gap': float(probs[top] - runner_up),
            'valid_movement': legal_movement,
            'valid_actions': int(valid.sum()),
        })
    return _original_act(self, role, obs, state, greedy=greedy)


def report():
    if not _probability_rows:
        print('[PROBS] No Routing decisions were observed.')
        return
    rows = _probability_rows
    n = len(rows)
    p0 = np.array([r['p0'] for r in rows])
    gap = np.array([r['gap'] for r in rows])
    counts = {str(a): sum(r['argmax'] == a for r in rows) for a in range(5)}
    with_moves = [r for r in rows if r['valid_movement']]
    p0_move = np.array([r['p0'] for r in with_moves])
    print('\n========== ROUTING POLICY PROBABILITY DIAGNOSTICS ==========')
    print(f'[PROBS] Routing decisions observed: {n}')
    print(f'[PROBS] Greedy argmax counts (0..4): {counts}')
    print(f'[PROBS] P(action 0): mean={p0.mean():.4f}, '
          f'median={np.median(p0):.4f}, min={p0.min():.4f}, max={p0.max():.4f}')
    print(f'[PROBS] P(action 0) >= 0.50: {(p0 >= .5).sum()}/{n} '
          f'({100*(p0 >= .5).mean():.1f}%)')
    print(f'[PROBS] P(action 0) >= 0.80: {(p0 >= .8).sum()}/{n} '
          f'({100*(p0 >= .8).mean():.1f}%)')
    print(f'[PROBS] Mean lead of winning action over runner-up: {gap.mean():.4f}')
    print(f'[PROBS] Decisions with >=1 legal move: {len(with_moves)}/{n}')
    if len(p0_move):
        print(f'[PROBS] P(action 0) when movement is legal: mean={p0_move.mean():.4f}')
    print('[PROBS] These are policy probabilities, not action frequencies.')
    print('============================================================\n')


if __name__ == '__main__':
    MAPPO.act = observed_act
    try:
        probe_part_flow.main()
    finally:
        report()
