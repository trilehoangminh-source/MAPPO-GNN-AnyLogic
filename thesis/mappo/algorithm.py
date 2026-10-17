"""Independent on-policy MAPPO optimization with role-specific centralized critics."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn

from .networks import Actor, CentralCritic


@dataclass
class MAPPOConfig:
    gamma: float = 0.90  # Tobias paper tested 0.90; tune with controls
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    epochs: int = 5
    mini_batches: int = 1
    hidden: int = 32
    max_grad_norm: float = 0.5
    value_normalization: bool = True
    normalize_advantage: bool = True
    routing_backbone: str = 'original_gnn'
    dispatch_backbone: str = 'original_linear'
    env_type: str = 'matrix'
    two_fleets: bool = True


class RunningValueNorm:
    def __init__(self, eps=1e-4):
        self.count = float(eps)
        self.mean = 0.0
        self.m2 = float(eps)

    @property
    def std(self):
        return max(1e-4, (self.m2 / self.count) ** 0.5)

    def update(self, targets):
        a = np.asarray(targets, dtype=np.float64)
        if not len(a):
            return
        n, mean, m2 = float(a.size), float(a.mean()), float(((a - a.mean()) ** 2).sum())
        delta = mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += m2 + delta * delta * self.count * n / total
        self.count = total

    def unnormalize(self, values):
        return values * self.std + self.mean

    def normalize(self, values):
        return (values - self.mean) / self.std

    def state_dict(self):
        return dict(mean=self.mean, m2=self.m2, count=self.count)

    def load_state_dict(self, d):
        self.mean, self.m2, self.count = d['mean'], d['m2'], d['count']


def as_actor(obs, device):
    return {k: torch.as_tensor(obs[k], dtype=torch.float32, device=device) for k in ('agvs', 'stat', 'action_mask')}


def as_global(state, device):
    return dict(agvs=torch.as_tensor(state['agvs'], dtype=torch.float32, device=device),
                stat=torch.as_tensor(state['stat'], dtype=torch.float32, device=device),
                ego=torch.as_tensor(state['ego'], dtype=torch.long, device=device))


def batchify(obs):
    return {k: v.unsqueeze(0) for k, v in obs.items()}


def stack_dict(items, converter, device):
    coll = [converter(item, device) for item in items]
    return {k: torch.stack([a[k] for a in coll]) for k in coll[0]}


class RolePolicy:
    def __init__(self, obs_space, feature_dim, role, action_dim, config, device):
        self.actor = Actor(obs_space, role, action_dim, config.routing_backbone if role == 'routing' else config.dispatch_backbone, hidden=config.hidden, env_type=config.env_type).to(device)
        self.critic = CentralCritic(feature_dim, config.hidden).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.valuenorm = RunningValueNorm()
        self.device = device

    @torch.no_grad()
    def act(self, obs, global_state, greedy=False):
        # BatchNorm in Tobias's feature extractors must run in eval mode for action selection.
        self.actor.eval()
        self.critic.eval()
        obs_t = batchify(as_actor(obs, self.device))
        global_t = batchify(as_global(global_state, self.device))
        dist = self.actor.dist(obs_t)
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        val = self.valuenorm.unnormalize(self.critic(global_t))
        return int(action.item()), float(logp.item()), float(val.item())


class MAPPO:
    def __init__(self, actor_spaces, global_feature_dim, cfg: MAPPOConfig, device='cpu'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.roles = {}
        for role_name, (obs_space, action_dim) in actor_spaces.items():
            role = 'routing' if role_name == 'routing' else 'dispatching'
            self.roles[role_name] = RolePolicy(obs_space, global_feature_dim, role, action_dim, cfg, self.device)

    @staticmethod
    def role_for(agent_id, ego, two_fleets=True):
        if int(agent_id) < 1000:
            return 'routing'
        if two_fleets:
            return 'dispatch_front' if ego % 2 == 0 else 'dispatch_rear'
        return 'dispatching'

    @torch.no_grad()
    def value(self, role, state):
        net = self.roles[role]
        net.critic.eval()
        val = net.critic(batchify(as_global(state, self.device)))
        return float(net.valuenorm.unnormalize(val).item())

    def act(self, role, obs, state, greedy=False):
        return self.roles[role].act(obs, state, greedy)

    def train(self, trajectories):
        """Trajectories: list[list[transition]] grouped by episode+physical agent.

        Each transition already contains the NEXT decision's accumulated reward,
        next_value and terminal (as emitted by AnyLogic). No fictitious turns.
        """
        grouped = defaultdict(list)
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda
        for trajectory in trajectories:
            gae = 0.0
            for step in reversed(trajectory):
                next_v = 0.0 if step['done'] else step['next_value']
                delta = step['reward'] + gamma * next_v - step['value']
                gae = delta + gamma * lam * (0.0 if step['done'] else gae)
                record = dict(step)
                record['advantage'] = gae
                record['return_target'] = gae + step['value']
                grouped[step['role']].append(record)
        report = {}
        for role, items in grouped.items():
            net = self.roles[role]
            obs = stack_dict([s['observation'] for s in items], as_actor, self.device)
            gs = stack_dict([s['state'] for s in items], as_global, self.device)
            actions = torch.tensor([s['action'] for s in items], dtype=torch.long, device=self.device)
            old_logp = torch.tensor([s['logp'] for s in items], dtype=torch.float32, device=self.device)
            adv = torch.tensor([s['advantage'] for s in items], dtype=torch.float32, device=self.device)
            returns = np.array([s['return_target'] for s in items], np.float64)
            if self.cfg.normalize_advantage and len(adv) > 1:
                adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp(min=1e-6)
            if self.cfg.value_normalization:
                net.valuenorm.update(returns)
                v_targets = torch.tensor(net.valuenorm.normalize(returns), dtype=torch.float32, device=self.device)
            else:
                v_targets = torch.tensor(returns, dtype=torch.float32, device=self.device)
            n = len(items)
            n_parts = max(1, min(self.cfg.mini_batches, n))
            net.actor.train()
            # Original Tobias encoders contain BatchNorm. Keep its running
            # statistics FIXED for both rollout and PPO optimization, otherwise
            # log pi_old (eval) and log pi_current (train) are not comparable.
            for layer in net.actor.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
            net.critic.train()
            total_actor = total_value = total_kl = 0.0
            updates = 0
            for _ in range(self.cfg.epochs):
                for idx in torch.tensor_split(torch.randperm(n, device=self.device), n_parts):
                    sub_obs = {k:v[idx] for k,v in obs.items()}
                    sub_gs = {k:v[idx] for k,v in gs.items()}
                    dist = net.actor.dist(sub_obs)
                    logp = dist.log_prob(actions[idx])
                    ratio = torch.exp(logp - old_logp[idx])
                    clipped = ratio.clamp(1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon)
                    actor_loss = -(torch.minimum(ratio*adv[idx], clipped*adv[idx])).mean()
                    entropy = dist.entropy().mean()
                    net.actor_optimizer.zero_grad()
                    (actor_loss - self.cfg.entropy_coef * entropy).backward()
                    nn.utils.clip_grad_norm_(net.actor.parameters(), self.cfg.max_grad_norm)
                    net.actor_optimizer.step()
                    v = net.critic(sub_gs)
                    value_loss = (v - v_targets[idx]).square().mean()
                    net.critic_optimizer.zero_grad()
                    (self.cfg.value_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(net.critic.parameters(), self.cfg.max_grad_norm)
                    net.critic_optimizer.step()
                    total_actor += actor_loss.item()
                    total_value += value_loss.item()
                    total_kl += (old_logp[idx] - logp.detach()).mean().item()
                    updates += 1
            report[role] = dict(transitions=n, actor_loss=total_actor/updates,
                                value_loss=total_value/updates, approx_kl=total_kl/updates,
                                optimizer_steps=updates)
        return report

    def save(self, path):
        torch.save(dict(config=asdict(self.cfg), roles={name:dict(
            actor=net.actor.state_dict(), critic=net.critic.state_dict(),
            actor_optimizer=net.actor_optimizer.state_dict(), critic_optimizer=net.critic_optimizer.state_dict(),
            valuenorm=net.valuenorm.state_dict()) for name,net in self.roles.items()}), path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        for name, state in data['roles'].items():
            net = self.roles[name]
            net.actor.load_state_dict(state['actor'])
            net.critic.load_state_dict(state['critic'])
            net.actor_optimizer.load_state_dict(state['actor_optimizer'])
            net.critic_optimizer.load_state_dict(state['critic_optimizer'])
            net.valuenorm.load_state_dict(state['valuenorm'])
