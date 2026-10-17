"""Event-driven Tobias/AnyLogic MAPPO collector with lifecycle-safe GAE segments.

Paste this file over thesis/mappo/collector.py after making a backup.

Policy sampling, action masks, reward matching, and public episode() API remain
unchanged. Reward for an action is consumed only upon the NEXT observation of
that same (role, physical AGV). A change of PettingZoo alias due to an AGV
leaving the system is a terminal boundary for that AGV's old trajectory.

Final actions of agents that are not called again before episode cutoff have
NO observed reward. They are reported and excluded, never assigned a fake 0.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np


class EventCollector:
    def __init__(self, model, environment, stop_mask_mode="baseline"):
        if stop_mask_mode not in ("baseline", "forbid_when_move"):
            raise ValueError(f"Unknown stop_mask_mode: {stop_mask_mode}")
        self.stop_mask_mode = stop_mask_mode
        self.model = model
        self.env = environment
        self.unfinished_last_episode = 0

    def _prepare(self, observation, role):
        mask = np.asarray(observation['action_mask'], dtype=np.float32).reshape(-1)
        n = self.model.roles[role].actor.action_dim
        if len(mask) < n:
            raise ValueError(f'{role}: mask has {len(mask)} entries; expected {n}')
        mask = mask[:5] if role == 'routing' else mask[:n]
        # B experiment: restrict Stop (action 0) ONLY when some real
        # movement action (1..4) was allowed by Tobias's original mask.
        # IMPORTANT: change mask BEFORE model.act(), and save it with the
        # transition; PPO update must use the same sampling distribution.
        mask = mask.copy()
        if (role == 'routing' and self.stop_mask_mode == 'forbid_when_move'
                and mask[0] > 0.5 and (mask[1:5] > 0.5).any()):
            mask[0] = 0.0
        if not (mask > 0.5).any():
            raise RuntimeError(f'{role}: no action allowed by environment mask')
        return {
            'agvs': np.asarray(observation['agvs'], dtype=np.float32).copy(),
            'stat': np.asarray(observation['stat'], dtype=np.float32).copy(),
            'action_mask': mask.copy(),
        }

    def _is_agent_replaced(self, role, info):
        """Match ONLY the conditions that Matrix._save_observation uses to
        replace a PettingZoo agent with a new alias for the same physical AGV.
        Do not confuse a whole-environment timeout with agent replacement.
        """
        if not isinstance(info, dict):
            return False
        if role == 'routing' and getattr(self.env, 'routing_agent_death', False):
            return (('in_system' in info and not bool(info['in_system']))
                    or ('at_target' in info and bool(info['at_target'])))
        if role != 'routing' and getattr(self.env, 'dispatching_agent_death', False):
            return (('in_system' in info and not bool(info['in_system']))
                    or ('at_target' in info and bool(info['at_target'])))
        return False

    def episode(self, seed=None, train=True, max_event_steps=None):
        self.env.reset(seed=seed)
        pending = {}  # (role, physical index) -> action awaiting its reward
        active = defaultdict(list)  # GAE segment for current lifetime only
        segments = []  # separate trajectories; NEVER concatenate lifetimes

        def finish_segment(key):
            steps = active.pop(key, None)
            if steps:
                segments.append(steps)

        events = 0
        delivered_reward = 0.0
        terminal = False
        alias_boundaries = Counter()
        action_histograms = defaultdict(Counter)
        stopped_with_move = 0
        stopped_without_move = 0
        routing_move_available_decisions = 0
        routing_stop_masked_decisions = 0
        mask_violations = 0
        ended_with = 'agents_exhausted'

        while self.env.agents:
            obs, reward, done, info = self.env.last()
            agent_id = str(self.env.agent_selection)
            state = self.env.get_mappo_state()
            physical = int(state['ego'])
            role = self.model.role_for(agent_id, physical, self.model.cfg.two_fleets)
            if role not in self.model.roles:
                raise RuntimeError(f'Unexpected role={role}, agent={agent_id}')
            key = (role, physical)
            sim_time = float(self.env.get_mappo_time())
            replaced = self._is_agent_replaced(role, info)

            if key in pending:
                previous = pending.pop(key)
                previous['reward'] = float(reward)
                # Aliased agents are NEW PettingZoo lives even if their
                # physical slot stays identical. Do not bootstrap across lives.
                previous['done'] = bool(done or replaced)
                previous['next_value'] = (
                    0.0 if previous['done'] else float(self.model.value(role, state))
                )
                previous['elapsed_seconds'] = float(sim_time - previous['time'])
                if previous['elapsed_seconds'] < -1e-5:
                    raise RuntimeError(f'AnyLogic time moved backward for {key}')
                active[key].append(previous)
                delivered_reward += float(reward)

            if replaced:
                alias_boundaries[role] += 1
                finish_segment(key)

            if done:
                terminal = True
                ended_with = 'environment_terminal'
                finish_segment(key)
                break

            if role == 'routing':
                original_mask = np.asarray(obs['action_mask'], dtype=np.float32).reshape(-1)[:5]
                move_available = bool((original_mask[1:5] > 0.5).any())
                routing_move_available_decisions += int(move_available)
                if (self.stop_mask_mode == 'forbid_when_move' and move_available
                        and original_mask[0] > 0.5):
                    routing_stop_masked_decisions += 1
            actor_obs = self._prepare(obs, role)
            action, logp, value = self.model.act(role, actor_obs, state, greedy=not train)
            action = int(action)
            if (not 0 <= action < len(actor_obs['action_mask'])
                    or actor_obs['action_mask'][action] <= 0.5):
                mask_violations += 1
                raise RuntimeError(f'{role} AGV={physical}: action {action} violates mask '
                                   f'{actor_obs["action_mask"].tolist()}')

            action_histograms[role][str(action)] += 1
            if role == 'routing' and action == 0:
                if (actor_obs['action_mask'][1:5] > 0.5).any():
                    stopped_with_move += 1
                else:
                    stopped_without_move += 1

            # This current decision becomes an action of the current/new
            # lifetime; it receives its reward only on its next observation.
            pending[key] = {
                'role': role, 'observation': actor_obs, 'state': state,
                'action': action, 'logp': float(logp), 'value': float(value),
                'time': sim_time,
            }
            self.env.step(action)
            events += 1
            if max_event_steps is not None and events >= max_event_steps:
                ended_with = 'max_event_steps'
                break

        # Non-terminal partial segments: retain their last available bootstrap
        # next_value; DO NOT synthesize a last reward for pending actions.
        for key in list(active):
            finish_segment(key)
        self.unfinished_last_episode = len(pending)
        statistics = dict(self.env.statistics or {})
        pending_by_role = Counter(role for role, _ in pending)

        return segments, {
            'events': events,
            'closed_transitions': sum(len(t) for t in segments),
            'discarded_pending': len(pending),
            'discarded_pending_by_role': dict(pending_by_role),
            'delivered_reward': delivered_reward,
            'reached_terminal': terminal,
            'end_reason': ended_with,
            'trajectories': len(segments),
            'alias_boundaries_by_role': dict(alias_boundaries),
            'statistics': statistics,
            'action_diagnostics': {
                'actions_by_role': {k: dict(v) for k, v in action_histograms.items()},
                'routing_stop_with_move_available': stopped_with_move,
                'routing_stop_without_move_available': stopped_without_move,
                'selected_mask_violations': mask_violations,
                'stop_mask_mode': self.stop_mask_mode,
                'routing_move_available_decisions': routing_move_available_decisions,
                'routing_stop_masked_decisions': routing_stop_masked_decisions,
            },
        }
