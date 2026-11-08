"""V14 Routing reward shaping at the AnyLogic observation -> Python reward boundary.

Inspired by (not an implementation of) Phan et al. (2026) and Ma & Lian (2023).
Only Routing MultiAgent behavior is affected. Dispatch reward, masks, and PPO
transition accounting remain unchanged. The original behaviors_v121.py is NOT
modified on disk: the new V14 experiment installs a process-local method wrapper.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from math import hypot, isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    stop_penalty: float = -0.5
    swing_penalty: float = -1.0
    candidate_clearance: float = 0.025
    local_risk_radius: float = 0.040
    position_tolerance: float = 0.002

    def __post_init__(self):
        if not all(isfinite(v) for v in asdict(self).values()):
            raise ValueError('Reward parameters must be finite')
        if self.stop_penalty > 0 or self.swing_penalty > 0:
            raise ValueError('Shaping terms MUST be non-positive')
        if not 0 < self.position_tolerance < self.candidate_clearance:
            raise ValueError('Invalid position_tolerance/candidate_clearance')
        if self.local_risk_radius < self.candidate_clearance:
            raise ValueError('local_risk_radius must not be smaller than candidate_clearance')

    def metadata(self) -> dict[str, float]:
        return asdict(self)


class RoutingRewardShaper:
    """State is keyed by physical AGV (0..19) and reset per episode.

    On the *next raw observation* for a physical Routing AGV, directly change
    the reward returned by MultiAgent.convert_from_observation(). We DO NOT
    rewrite completed trajectories or try to inject reward into arbitrary PPO
    transitions. The normal existing collector still owns reward bookkeeping.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.reset_episode()
        self._installed_on = None

    def reset_episode(self):
        self._states: dict[int, dict[str, Any]] = {}
        self.counts = Counter()
        self.total_stop_penalty = 0.0
        self.total_swing_penalty = 0.0

    @staticmethod
    def _xy(row: np.ndarray, start: int) -> tuple[float, float] | None:
        if len(row) < start + 2:
            return None
        x, y = float(row[start]), float(row[start + 1])
        if not (isfinite(x) and isfinite(y)):
            return None
        return x, y

    @staticmethod
    def _near(a: tuple[float, float], b: tuple[float, float], radius: float) -> bool:
        return hypot(a[0] - b[0], a[1] - b[1]) <= radius

    def _apparently_free_move(self, obs: dict) -> tuple[bool, str]:
        """Conservative proxy, not proof that waiting is unnecessary.

        A movement must be allowed by the ORIGINAL routing mask, its neighbor
        node must be present, no other AGV may be locally close to the ego, and
        a candidate node must be clear of others' current/waypoint coordinates.
        If necessary data are missing, do not penalize the wait.
        """
        agvs = np.asarray(obs['agvs'], dtype=float)
        mask = np.asarray(obs['action_mask'], dtype=float).reshape(-1)
        if agvs.ndim != 2 or agvs.shape[0] < 1 or agvs.shape[1] < 16 or len(mask) < 5:
            return False, 'missing_observation'
        ego = agvs[0]
        if not np.isfinite(ego).all() or float(ego[0]) < 0.5:
            return False, 'inactive_or_invalid_ego'
        current = self._xy(ego, 2)
        if current is None:
            return False, 'missing_position'
        allowed = [i for i in range(1, 5) if mask[i] > 0.5]
        if not allowed:
            return False, 'no_legal_move'
        others = [row for row in agvs[1:] if len(row) >= 6 and
                  np.isfinite(row).all() and row[0] > 0.5]
        # Locally busy junction: yield is considered potentially necessary.
        for row in others:
            other_current = self._xy(row, 2)
            if other_current is not None and self._near(current, other_current,
                                                       self.config.local_risk_radius):
                return False, 'nearby_agv'
        for action in allowed:
            node = self._xy(ego, 8 + (action - 1) * 2)
            if node is None or (abs(node[0]) <= 1e-12 and abs(node[1]) <= 1e-12):
                continue
            # A node is suspect if another AGV occupies it or appears to head
            # for it. Fields 4:6 are only an observed waypoint proxy.
            contested = False
            for row in others:
                for offset in (2, 4):
                    other_xy = self._xy(row, offset)
                    if other_xy is not None and self._near(node, other_xy,
                                                           self.config.candidate_clearance):
                        contested = True
                        break
                if contested:
                    break
            if not contested:
                return True, 'apparently_clear_candidate'
        return False, 'no_clear_candidate'

    def observe(self, physical: int, obs: dict, raw_reward: float) -> float:
        """Called IN convert_from_observation (not after PPO rollout)."""
        physical = int(physical)
        st = self._states.setdefault(physical, {
            'pending_action': None,
            'pending_stop_penalty': False,
            'last_observation': None,
            'positions': deque(maxlen=2),
        })
        self.counts['routing_observations'] += 1
        reward = float(raw_reward)
        pending = st['pending_action']
        agvs = np.asarray(obs['agvs'], dtype=float)
        ego_active = (agvs.ndim == 2 and bool(len(agvs)) and
                      agvs.shape[1] >= 4 and np.isfinite(agvs[0]).all() and
                      agvs[0, 0] > 0.5)
        if pending == 0 and st['pending_stop_penalty'] and ego_active:
            reward += self.config.stop_penalty
            self.total_stop_penalty += self.config.stop_penalty
            self.counts['stop_penalties_applied'] += 1
        st['pending_action'] = None
        st['pending_stop_penalty'] = False

        if ego_active:
            position = self._xy(agvs[0], 2)
            if position is not None:
                history = st['positions']
                if not history or not self._near(position, history[-1], self.config.position_tolerance):
                    # Distinct routing-node observations: A -> B -> A is a swing.
                    if (pending is not None and pending != 0 and len(history) == 2
                            and self._near(position, history[0], self.config.position_tolerance)
                            and not self._near(position, history[1], self.config.position_tolerance)):
                        reward += self.config.swing_penalty
                        self.total_swing_penalty += self.config.swing_penalty
                        self.counts['swing_penalties_applied'] += 1
                    history.append(position)
        st['last_observation'] = obs
        return reward

    def choose(self, physical: int, action: int):
        """Called from the existing Routing convert_to_action method."""
        physical = int(physical)
        action = int(action)
        st = self._states.setdefault(physical, {
            'pending_action': None,
            'pending_stop_penalty': False,
            'last_observation': None,
            'positions': deque(maxlen=2),
        })
        st['pending_action'] = action
        st['pending_stop_penalty'] = False
        self.counts['routing_decisions'] += 1
        if action != 0:
            self.counts['move_decisions'] += 1
            return
        self.counts['stop_decisions'] += 1
        if st['last_observation'] is None:
            self.counts['protected_wait_missing_observation'] += 1
            return
        eligible, reason = self._apparently_free_move(st['last_observation'])
        if eligible:
            st['pending_stop_penalty'] = True
            self.counts['apparently_unnecessary_stops_selected'] += 1
        else:
            self.counts[f'protected_wait_{reason}'] += 1

    def report(self) -> dict[str, Any]:
        d = dict(self.counts)
        d.update({
            'mode': 'direct_behavior_reward_v14',
            'configuration': self.config.metadata(),
            'total_stop_penalty': self.total_stop_penalty,
            'total_swing_penalty': self.total_swing_penalty,
            'total_reward_adjustment': self.total_stop_penalty + self.total_swing_penalty,
            'note': 'Routing-only observation-time shaping; free-move/necessary-wait classification is heuristic',
        })
        return d

    def install(self, module=None):
        """Wrap actual V14 environment behavior methods; no source overwrite."""
        if module is None:
            from thesis.envs import behaviors_v121 as module
        cls = module.MultiAgent  # V14 routes through MultiAgent, not SingleAgent.
        if getattr(cls, '_v14_reward_wrapper_installed', False):
            raise RuntimeError('V14 reward wrapper already installed in this process')
        orig_obs = cls.convert_from_observation
        orig_act = cls.convert_to_action
        shaper = self

        def with_reward(instance, alpyne_obs):
            converted = orig_obs(instance, alpyne_obs)
            # Never touch Dispatch Front/Rear or pseudorouting agents.
            caller = int(alpyne_obs.caller)
            if getattr(instance, 'dispatching', False) or caller < 0 or caller >= 1000:
                return converted
            obs, reward, done, info = converted
            return obs, shaper.observe(caller % 1000, obs, reward), done, info

        def with_decision(instance, actions, agent):
            raw_action = actions[0] if isinstance(actions, (list, tuple)) else actions
            alpyne_action = orig_act(instance, actions, agent)
            physical = int(agent)
            if not getattr(instance, 'dispatching', False) and 0 <= physical < 1000:
                shaper.choose(physical % 1000, 0 if raw_action is None else int(raw_action))
            return alpyne_action

        cls.convert_from_observation = with_reward
        cls.convert_to_action = with_decision
        cls._v14_reward_wrapper_installed = True
        self._installed_on = cls


def default_shaping_config() -> RewardConfig:
    return RewardConfig()
