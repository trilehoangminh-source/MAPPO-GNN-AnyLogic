#======================= GIẢI THÍCH FILE: GAE VÀ PPO ỔN ĐỊNH V7 =======================
# Tính GAE/return, ValueNorm, clipped policy/value loss và xác minh log-prob rollout.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================


"""V7 MAPPO stability upgrades for event-driven Tobias/AnyLogic.

A non-destructive replacement of `MAPPO.train` on a single model instance.
Keeps actor, critic, routing Dijkstra prior, dispatching actor and checkpoints
unchanged. Inspired by marlbenchmark/on-policy (R_MAPPO), independently adapted.
"""



#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, asdict
from types import MethodType
import math
import os
import numpy as np
import torch
from torch import nn
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================




#============================= MODULE 02 - LỚP V7OPTIONS ==============================
# Bộ tùy chọn ổn định PPO: value clipping, KL threshold, discount thời gian.
@dataclass(frozen=True)
class V7Options:
    value_loss: str = 'clipped_huber'  # mse / clipped_mse / clipped_huber
    value_clip_epsilon: float = 0.2
    huber_delta: float = 1.0
    target_kl: float = 0.03  # positive approximate KL stop threshold, 0 disables
    time_unit_seconds: float = 0.0  # 0: classic per-decision GAE, >0: experimental SMDP GAE



    #========================= MODULE CON 01 - __POST_INIT__ ==========================
    # Mục đích: Kiểm tra giá trị tham số sau khi tạo dataclass.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __post_init__(self):
        if self.value_loss not in ('mse', 'clipped_mse', 'clipped_huber'):
            raise ValueError('value_loss must be mse/clipped_mse/clipped_huber')
        if not math.isfinite(self.value_clip_epsilon) or self.value_clip_epsilon <= 0:
            raise ValueError('value_clip_epsilon must be > 0')
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError('huber_delta must be > 0')
        if not math.isfinite(self.target_kl) or self.target_kl < 0:
            raise ValueError('target_kl must be >= 0')
        if not math.isfinite(self.time_unit_seconds) or self.time_unit_seconds < 0:
            raise ValueError('time_unit_seconds must be >= 0')
    #===================== KẾT THÚC MODULE CON 01 - __POST_INIT__ =====================

#========================= KẾT THÚC MODULE 02 - LỚP V7OPTIONS =========================




#========================== MODULE 03 - HÀM MAKE_GAE_RECORDS ==========================
# Duyệt ngược trajectory, tính delta_t và GAE; return_target = advantage + V(s).
def make_gae_records(trajectories, gamma: float, lam: float, time_unit_seconds: float = 0.):
    """One trajectory per physical AGV lifetime; NO stitching across aliases.

    In the optional elapsed-time experiment, gamma_t=gamma**(dt/time_unit).
    Lambda still applies *per decision*: gamma_t * lambda. Training mode must
    be compared as an ablation; it does not reconstruct undisclosed rewards.
    """

    #====================== MỤC NHỎ - NHÓM TRAJECTORY THEO ROLE =======================
    # Trajectories chia theo lifetime; GAE không ghép qua các alias.
    groups = defaultdict(list)
    for trajectory in trajectories:
        gae = 0.0

        #===================== MỤC NHỎ - TÍNH GAE NGƯỢC THỜI GIAN =====================
        # delta = r + gamma_t V_next - V; A = delta + gamma_t lambda A_next.
        for step in reversed(trajectory):
            if time_unit_seconds > 0:
                if 'elapsed_seconds' not in step:
                    raise RuntimeError('SMDP GAE requires elapsed_seconds on each transition')
                dt = float(step['elapsed_seconds'])
                if not math.isfinite(dt) or dt < 0:
                    raise ValueError('Invalid elapsed_seconds in GAE')
                gamma_t = float(gamma ** (dt / time_unit_seconds))
            else:
                gamma_t = float(gamma)
            done = bool(step['done'])
            nxt = 0.0 if done else float(step['next_value'])
            delta = float(step['reward']) + gamma_t * nxt - float(step['value'])
            gae = delta + gamma_t * float(lam) * (0.0 if done else gae)
            if not math.isfinite(gae):
                raise FloatingPointError('Nonfinite advantage; inspect rewards and values')
            entry = dict(step)
            entry['advantage'] = gae

            #====================== MỤC NHỎ - TẠO TARGET CRITIC =======================
            # Return target theo thang reward gốc = GAE + giá trị rollout.
            entry['return_target'] = gae + float(step['value'])
            groups[step['role']].append(entry)
    return groups
#===================== KẾT THÚC MODULE 03 - HÀM MAKE_GAE_RECORDS ======================




#============================ MODULE 04 - HÀM CRITIC_LOSS =============================
# Tính giá trị lỗi Critic có thể clip và/hoặc dùng Huber, trên thang đã ValueNorm.
def critic_loss(predicted, old_value_predictions, normalized_targets, config: V7Options):
    """PPO clipped value loss in CURRENT value-normalizer coordinates.

    `old_value_predictions` are rollout critic predictions, normalized using
    the SAME snapshot of RunningValueNorm as current regression targets.
    """
    if config.value_loss == 'clipped_huber':
        def err_fn(err):
            return torch.nn.functional.smooth_l1_loss(
                err, torch.zeros_like(err), beta=config.huber_delta, reduction='none')
    else:
        def err_fn(err):
            return err.square()
    original = err_fn(predicted - normalized_targets)
    if config.value_loss == 'mse':
        return original.mean()
    clipped = old_value_predictions + torch.clamp(
        predicted - old_value_predictions,
        -config.value_clip_epsilon, config.value_clip_epsilon)
    return torch.maximum(original, err_fn(clipped - normalized_targets)).mean()
#======================== KẾT THÚC MODULE 04 - HÀM CRITIC_LOSS ========================




#============================== MODULE 05 - HÀM TRAIN_V7 ==============================
# PPO ổn định cho từng role: advantage, ratio/clipping, entropy, Actor/Critic Adam.
def train_v7(self, trajectories):
    """Train role-specific actor/centralized critic with strict on-policy masks.

    This method is attached only to the V7 model instance by enable_v7().
    Original GNN/MLP/Dijkstra functions and MAPPO.train remain untouched.
    """
    from thesis.mappo import algorithm as algo

    cfg = self.cfg
    opts = self._v7_options

    #======================== MỤC NHỎ - TÍNH ADVANTAGE/RETURN =========================
    # Phân chia trajectory và tính GAE theo role.
    grouped = make_gae_records(trajectories, cfg.gamma, cfg.gae_lambda,
                               opts.time_unit_seconds)
    reports = {}

    #========================= MỤC NHỎ - HUẤN LUYỆN TỪNG ROLE =========================
    # Mỗi role có Actor, Critic, ValueNorm và Adam riêng.
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
        # Physical Routing policy uses 5 logits with NO action mask (Tobias
        # joint experiment). The environment's mask is diagnostic for Routing,
        # not a constraint on the sampling distribution. All other policies
        # continue to enforce their masks exactly as before.
        n_actions = int(obs['action_mask'].shape[-1])
        if bool((actions < 0).any()) or bool((actions >= n_actions).any()):
            raise RuntimeError(f'{role}: rollout contains out-of-range action')
        if getattr(net.actor, 'uses_action_mask', True):
            legal = obs['action_mask'].gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            if bool((legal <= 0.5).any()):
                raise RuntimeError('Rollout includes an action forbidden by stored action mask')


        #======================= MỤC NHỎ - CHUẨN HÓA ADVANTAGE ========================
        # Chuẩn hóa advantage, cập nhật ValueNorm và chuẩn hóa targets Critic.
        if cfg.normalize_advantage and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp(min=1e-6)
        # Update normalizer ONCE per rollout, never once per PPO epoch/minibatch.
        if cfg.value_normalization:
            net.valuenorm.update(returns)
            normalized_returns = net.valuenorm.normalize(returns)
            normalized_old_values = net.valuenorm.normalize(old_values)
        else:
            normalized_returns, normalized_old_values = returns, old_values
        targets = torch.as_tensor(normalized_returns, dtype=torch.float32, device=self.device)
        old_vpred = torch.as_tensor(normalized_old_values, dtype=torch.float32, device=self.device)


        #==================== MỤC NHỎ - ĐƯA MẠNG VÀO CHẾ ĐỘ TRAIN =====================
        # Giữ thống kê BatchNorm ổn định giữa rollout và update.
        net.actor.train()
        # Rollout occurs in eval() mode. Preserve BatchNorm statistics and
        # distributions in PPO to avoid changing policy *without an update*.
        for layer in net.actor.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        net.critic.train()
        n = len(items)
        n_parts = max(1, min(cfg.mini_batches, n))
        # Optional memory safety guard for 4GB CUDA GPUs. The default remains
        # byte-for-byte equivalent to the original PPO minibatch partition.
        # Smaller minibatches add optimizer steps; compare resulting policies
        # by evaluation instead of assuming identical PPO update dynamics.
        max_mini = int(os.environ.get('MAPPO_MAX_MINIBATCH', '0'))
        if max_mini < 0:
            raise ValueError('MAPPO_MAX_MINIBATCH must be >= 0')
        if max_mini:
            n_parts = max(n_parts, math.ceil(n / max_mini))
            if role == 'routing':
                print(f'[V8 VRAM GUARD] routing_samples={n} minibatches={n_parts} max_samples_per_minibatch={math.ceil(n/n_parts)}', flush=True)
        sums = defaultdict(float)
        steps = 0
        epochs_done = 0
        early_stopped = False
        for epoch in range(cfg.epochs):
            for idx in torch.tensor_split(torch.randperm(n, device=self.device), n_parts):
                if idx.numel() == 0:
                    continue
                sub_obs = {key: val[idx] for key, val in obs.items()}
                sub_gs = {key: val[idx] for key, val in gs.items()}
                dist = net.actor.dist(sub_obs)
                new_logp = dist.log_prob(actions[idx])
                logratio = new_logp - old_logp[idx]
                if not torch.isfinite(logratio).all():
                    raise FloatingPointError('Nonfinite policy log ratio')

                #=================== MỤC NHỎ - PPO IMPORTANCE RATIO ===================
                # ratio = exp(log_pi_new - log_pi_old).
                ratio = torch.exp(logratio)
                if not torch.isfinite(ratio).all():
                    raise FloatingPointError('Nonfinite policy ratio')
                with torch.no_grad():
                    # Nonnegative sampled estimator; preferable for early stopping
                    # to mean(old_logp-new_logp), which may be negative.
                    approx_kl = ((ratio - 1.) - logratio).mean()
                    if opts.target_kl > 0 and approx_kl.item() > opts.target_kl:
                        early_stopped = True
                        break
                    clipfrac = ((ratio - 1.).abs() > cfg.clip_epsilon).float().mean()

                #================== MỤC NHỎ - CLIPPED SURROGATE LOSS ==================
                # Dùng min giữa ratio*A và clip(ratio)*A; cộng entropy bonus.
                surrogate = torch.minimum(
                    ratio * advantage[idx],
                    ratio.clamp(1. - cfg.clip_epsilon, 1. + cfg.clip_epsilon) * advantage[idx])
                actor_loss = -surrogate.mean()
                entropy = dist.entropy().mean()
                net.actor_optimizer.zero_grad()
                (actor_loss - cfg.entropy_coef * entropy).backward()
                a_norm = nn.utils.clip_grad_norm_(net.actor.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(a_norm):
                    raise FloatingPointError('Nonfinite actor gradient norm')
                net.actor_optimizer.step()


                #==================== MỤC NHỎ - CRITIC LOSS / ADAM ====================
                # Tính value loss riêng rồi backward/Adam cho Critic.
                predicted = net.critic(sub_gs)
                value_loss = critic_loss(predicted, old_vpred[idx], targets[idx], opts)
                net.critic_optimizer.zero_grad()
                (cfg.value_coef * value_loss).backward()
                c_norm = nn.utils.clip_grad_norm_(net.critic.parameters(), cfg.max_grad_norm)
                if not torch.isfinite(c_norm):
                    raise FloatingPointError('Nonfinite critic gradient norm')
                net.critic_optimizer.step()
                sums['actor_loss'] += float(actor_loss.detach())
                sums['value_loss'] += float(value_loss.detach())
                sums['approx_kl'] += float(approx_kl)
                sums['clip_fraction'] += float(clipfrac)
                sums['entropy'] += float(entropy.detach())
                sums['actor_grad_norm'] += float(a_norm)
                sums['critic_grad_norm'] += float(c_norm)
                steps += 1
            if early_stopped:
                break
            epochs_done = epoch + 1
        report = {k: sums[k] / max(steps, 1) for k in
                  ('actor_loss', 'value_loss', 'approx_kl', 'clip_fraction', 'entropy',
                   'actor_grad_norm', 'critic_grad_norm')}
        report.update(transitions=n, optimizer_steps=steps,
                      epochs_completed=epochs_done, early_stopped=early_stopped,
                      value_loss_mode=opts.value_loss,
                      discount_mode=('elapsed_seconds' if opts.time_unit_seconds else 'per_decision'))
        reports[role] = report
    return reports
#========================= KẾT THÚC MODULE 05 - HÀM TRAIN_V7 ==========================




#============================= MODULE 06 - HÀM ENABLE_V7 ==============================
# Gắn trainer V7 vào model instance, không sửa class MAPPO gốc.
def enable_v7(model, options: V7Options | None = None):
    """Install V7 update on only one MAPPO instance, leaving old versions alone."""
    if getattr(model, '_v7_stability_installed', False):
        raise RuntimeError('V7 MAPPO already installed on this model')
    if not hasattr(model, 'roles') or not hasattr(model, 'cfg'):
        raise TypeError('Expected MAPPO model with roles and cfg')
    model._v7_options = options or V7Options()
    model.train = MethodType(train_v7, model)
    model._v7_stability_installed = True
    return asdict(model._v7_options)
#========================= KẾT THÚC MODULE 06 - HÀM ENABLE_V7 =========================




#======================== MODULE 07 - HÀM AUDIT_ROLLOUT_ROLES =========================
# So sánh log-prob đã lưu khi rollout với log-prob mạng hiện tại trước PPO.
@torch.no_grad()
def audit_rollout_roles(model, trajectories, max_per_role: int = 32, atol: float = 1e-4):
    """Pre-update logp parity of ALL roles (Routing AND Dispatching)."""
    from thesis.mappo import algorithm as algo
    report = {}
    grouped = defaultdict(list)
    for segment in trajectories:
        for step in segment:
            role = step['role']
            if len(grouped[role]) < max_per_role:
                grouped[role].append(step)
    for role, steps in grouped.items():
        actor = model.roles[role].actor
        was_training = actor.training
        actor.eval()
        errors = []
        try:
            for step in steps:
                obs = step['observation']
                act = int(step['action'])
                mask = np.asarray(obs['action_mask']).reshape(-1)
                if act < 0 or act >= len(mask):
                    raise RuntimeError(f'{role}: out-of-range historical action {act}')
                # Routing with a no-mask actor is allowed to sample any of
                # its 5 logits. Dispatch and masked Routing must be legal.
                if (getattr(actor, 'uses_action_mask', True)
                        and mask[act] <= 0.5):
                    raise RuntimeError(f'{role}: invalid historical action {act}')
                input_t = algo.batchify(algo.as_actor(obs, model.device))
                dist = actor.dist(input_t)
                lp = float(dist.log_prob(torch.tensor([act], device=model.device)).item())
                errors.append(abs(lp - float(step['logp'])))
        finally:
            actor.train(was_training)
        err = max(errors, default=0.0)
        report[role] = dict(compared=len(errors), max_error=err, passed=err <= atol)
        if err > atol:
            raise RuntimeError(f'PPO pre-update logp mismatch in {role}: {err}')
    return report
#==================== KẾT THÚC MODULE 07 - HÀM AUDIT_ROLLOUT_ROLES ====================


