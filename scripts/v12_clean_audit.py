"""Pre-update PPO parity for V12 (NO Dijkstra action guidance).

V11's gnn_audit_uoe_v2.verify_ppo_rollout_parity requires a
`routing_guide_bias` on every Routing transition. V12 deliberately drops it.

This V12-only replacement retains the *actual* safety checks:
  * no forbidden action under the stored action mask;
  * stored action log-probability matches recomputation from the SAME policy;
  * finite probabilities, non-empty tested Routing transitions;
  * no legacy shortest-path bias accidentally surviving in the PPO buffer.

Never call this after a PPO optimizer step.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def verify_v12_ppo_rollout_parity(model, trajectories, *, max_samples: int = 64,
                                   atol: float = 1e-4):
    role = model.roles['routing']
    actor = role.actor
    samples = [step for segment in trajectories for step in segment
               if step.get('role') == 'routing']
    if not samples:
        raise RuntimeError('V12 PPO preflight: no Routing samples to validate')
    selected = samples[:max(1, int(max_samples))]
    was_training = actor.training
    actor.eval()
    errors = []
    mask_bad = 0
    try:
        with torch.no_grad():
            for step in selected:
                obs = step['observation']
                if 'routing_guide_bias' in obs:
                    raise RuntimeError('V12 PPO buffer unexpectedly contains legacy Dijkstra bias')
                action = int(step['action'])
                mask = np.asarray(obs['action_mask'], dtype=np.float32).reshape(-1)
                if (mask.size != 5 or not np.isfinite(mask).all() or
                        action < 0 or action >= mask.size or mask[action] <= 0.5):
                    mask_bad += 1
                    continue
                # Equivalent to algorithm.as_actor/batchify for a no-bias V12 obs;
                # no hidden global state or future features can enter the Actor.
                t_obs = {key: torch.as_tensor(obs[key], dtype=torch.float32,
                                              device=model.device).unsqueeze(0)
                         for key in ('agvs', 'stat', 'action_mask')}
                logp = role.actor.dist(t_obs).log_prob(
                    torch.tensor([action], dtype=torch.long, device=model.device))
                recomputed = float(logp.item())
                saved = float(step['logp'])
                if not math.isfinite(recomputed) or not math.isfinite(saved):
                    raise RuntimeError('V12 PPO preflight: nonfinite old/new log-probability')
                errors.append(abs(recomputed - saved))
    finally:
        actor.train(was_training)
    worst = max(errors, default=float('inf'))
    report = dict(routing_samples=len(selected), compared=len(errors),
                  mask_violations=mask_bad, max_logprob_error=worst,
                  tolerance=atol, passed=(mask_bad == 0 and
                                          len(errors) == len(selected) and
                                          worst <= atol),
                  dijkstra_guidance_required=False)
    if not report['passed']:
        raise RuntimeError('STOP before PPO update: V12 rollout/current logprob mismatch: '
                           + str(report))
    return report
