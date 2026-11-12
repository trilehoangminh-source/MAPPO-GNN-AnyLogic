"""Full-rollout PPO with VRAM-bounded gradient accumulation for V15.

One effective minibatch per role per PPO epoch. Microbatches are a memory
implementation detail: global advantage normalization, one ValueNorm update,
one weighted gradient sum, one gradient clipping and one optimizer step per epoch.
No change to the original MAPPO optimizer or V7 train function.
"""
from __future__ import annotations

from collections import defaultdict
from types import MethodType
import math
import numpy as np
import torch
from torch import nn

from thesis.mappo.mappo_stability_v7 import make_gae_records, critic_loss


def enable_fullrollout(model, microbatch_size: int = 512):
    """Replace only this model instance's V7 training method, after enable_v7()."""
    if not isinstance(microbatch_size, int) or microbatch_size < 1:
        raise ValueError('microbatch_size must be a positive integer')
    if not getattr(model, '_v7_stability_installed', False):
        raise RuntimeError('V7 stability must be installed before full-rollout accumulation')
    if int(model.cfg.mini_batches) != 1:
        raise ValueError('Full-rollout mode requires blueprint mini_batches=1')
    if getattr(model, '_v15_fullrollout_installed', False):
        raise RuntimeError('Full-rollout optimizer already installed')
    model._fullrollout_microbatch_size = microbatch_size
    model.train = MethodType(train_fullrollout, model)
    model._v15_fullrollout_installed = True
    return {'optimizer_mode': 'full_rollout_gradient_accumulation',
            'effective_minibatches_per_epoch': 1,
            'microbatch_size': microbatch_size}


def train_fullrollout(self, trajectories):
    from thesis.mappo import algorithm as algo

    cfg = self.cfg
    opts = self._v7_options
    microbatch_size = int(self._fullrollout_microbatch_size)
    grouped = make_gae_records(trajectories, cfg.gamma, cfg.gae_lambda,
                               opts.time_unit_seconds)
    reports = {}
    for role, items in grouped.items():
        if role not in self.roles:
            raise KeyError('Unknown role in trajectory: ' + str(role))
        net = self.roles[role]
        obs = algo.stack_dict([s['observation'] for s in items], algo.as_actor, self.device)
        gs = algo.stack_dict([s['state'] for s in items], algo.as_global, self.device)
        actions = torch.as_tensor([s['action'] for s in items], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor([s['logp'] for s in items], dtype=torch.float32, device=self.device)
        advantage = torch.as_tensor([s['advantage'] for s in items], dtype=torch.float32, device=self.device)
        returns = np.asarray([s['return_target'] for s in items], dtype=np.float64)
        old_values = np.asarray([s['value'] for s in items], dtype=np.float64)
        if not (torch.isfinite(old_logp).all() and torch.isfinite(advantage).all()
                and np.isfinite(returns).all() and np.isfinite(old_values).all()):
            raise FloatingPointError('Nonfinite on-policy rollout fields')
        n_actions = int(obs['action_mask'].shape[-1])
        if bool((actions < 0).any()) or bool((actions >= n_actions).any()):
            raise RuntimeError(f'{role}: rollout contains out-of-range action')
        if getattr(net.actor, 'uses_action_mask', True):
            legal = obs['action_mask'].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            if bool((legal <= 0.5).any()):
                raise RuntimeError('Rollout includes an action forbidden by stored action mask')

        # Both statistics are computed on the FULL rollout, never a microbatch.
        if cfg.normalize_advantage and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp(min=1e-6)
        if cfg.value_normalization:
            net.valuenorm.update(returns)
            normalized_returns = net.valuenorm.normalize(returns)
            normalized_old_values = net.valuenorm.normalize(old_values)
        else:
            normalized_returns, normalized_old_values = returns, old_values
        targets = torch.as_tensor(normalized_returns, dtype=torch.float32, device=self.device)
        old_vpred = torch.as_tensor(normalized_old_values, dtype=torch.float32, device=self.device)

        net.actor.train()
        # Match V7 behavior and maintain consistent distributions across
        # microbatches: do not update BatchNorm statistics during PPO training.
        for layer in net.actor.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        net.critic.train()
        n = len(items)
        n_micro = math.ceil(n / microbatch_size)
        print(f'[V15 FULL ROLLOUT] role={role} samples={n} '
              f'effective_minibatches=1 microbatches={n_micro} '
              f'microbatch_size={microbatch_size}', flush=True)

        sums = defaultdict(float)
        steps = 0
        microbatches_processed = 0
        early_stopped = False
        last_checked_kl = float('nan')
        for epoch in range(cfg.epochs):
            # A single shuffle per PPO epoch; no optimizer step within the loop.
            perm = torch.randperm(n, device=self.device)
            net.actor_optimizer.zero_grad(set_to_none=True)
            net.critic_optimizer.zero_grad(set_to_none=True)
            epoch_sums = defaultdict(float)

            for idx in perm.split(microbatch_size):
                part = idx.numel()
                if not part:
                    continue
                weight = part / n  # Important for an uneven final microbatch.
                sub_obs = {key: val[idx] for key, val in obs.items()}
                sub_gs = {key: val[idx] for key, val in gs.items()}
                dist = net.actor.dist(sub_obs)
                new_logp = dist.log_prob(actions[idx])
                logratio = new_logp - old_logp[idx]
                if not torch.isfinite(logratio).all():
                    raise FloatingPointError('Nonfinite policy log ratio')
                ratio = torch.exp(logratio)
                if not torch.isfinite(ratio).all():
                    raise FloatingPointError('Nonfinite policy ratio')
                with torch.no_grad():
                    kl = ((ratio - 1.) - logratio).mean()
                    clipfrac = ((ratio - 1.).abs() > cfg.clip_epsilon).float().mean()
                surrogate = torch.minimum(
                    ratio * advantage[idx],
                    ratio.clamp(1. - cfg.clip_epsilon, 1. + cfg.clip_epsilon) * advantage[idx])
                actor_loss = -surrogate.mean()
                entropy = dist.entropy().mean()
                # One weighted gradient sum over ALL samples before optimizer.step().
                (weight * (actor_loss - cfg.entropy_coef * entropy)).backward()

                predicted = net.critic(sub_gs)
                value_loss = critic_loss(predicted, old_vpred[idx], targets[idx], opts)
                (weight * cfg.value_coef * value_loss).backward()
                epoch_sums['actor_loss'] += weight * float(actor_loss.detach())
                epoch_sums['value_loss'] += weight * float(value_loss.detach())
                epoch_sums['approx_kl'] += weight * float(kl)
                epoch_sums['clip_fraction'] += weight * float(clipfrac)
                epoch_sums['entropy'] += weight * float(entropy.detach())
                microbatches_processed += 1

            # KL early-stop is based on the WHOLE rollout at the same weights.
            # If exceeded, discard all accumulated gradients (no partial update).
            last_checked_kl = epoch_sums['approx_kl']
            if opts.target_kl > 0 and last_checked_kl > opts.target_kl:
                net.actor_optimizer.zero_grad(set_to_none=True)
                net.critic_optimizer.zero_grad(set_to_none=True)
                early_stopped = True
                break

            a_norm = nn.utils.clip_grad_norm_(net.actor.parameters(), cfg.max_grad_norm)
            c_norm = nn.utils.clip_grad_norm_(net.critic.parameters(), cfg.max_grad_norm)
            if not (torch.isfinite(a_norm) and torch.isfinite(c_norm)):
                raise FloatingPointError('Nonfinite full-rollout gradient norm')
            net.actor_optimizer.step()
            net.critic_optimizer.step()
            epoch_sums['actor_grad_norm'] = float(a_norm)
            epoch_sums['critic_grad_norm'] = float(c_norm)
            for key, value in epoch_sums.items():
                sums[key] += value
            steps += 1

        report = {key: sums[key] / max(steps, 1) for key in
                  ('actor_loss', 'value_loss', 'approx_kl', 'clip_fraction', 'entropy',
                   'actor_grad_norm', 'critic_grad_norm')}
        report.update(transitions=n, optimizer_steps=steps,
                      epochs_completed=steps, early_stopped=early_stopped,
                      value_loss_mode=opts.value_loss,
                      discount_mode=('elapsed_seconds' if opts.time_unit_seconds else 'per_decision'),
                      optimizer_mode='full_rollout_gradient_accumulation',
                      effective_minibatches_per_epoch=1,
                      microbatch_size=microbatch_size,
                      microbatches_processed=microbatches_processed,
                      last_checked_full_rollout_kl=last_checked_kl)
        reports[role] = report
    return reports
