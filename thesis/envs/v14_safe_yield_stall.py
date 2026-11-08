"""V14 direct Routing reward at the environment observation boundary.

Experimental, conservative congestion-proxy shaping. Original AnyLogic reward,
Routing/Dispatch action masks and PPO collection logic are NOT modified.

Implementation hooks existing Matrix._save_observation *after* it has captured
physical-indexed AGVs and model time, but before the PettingZoo collector calls
last(). No completed PPO transitions are edited. Source .py files are not overwritten.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, asdict
from math import hypot, isfinite
from typing import Any
import numpy as np


@dataclass(frozen=True)
class SafeYieldConfig:
    yield_bonus: float = 0.35
    stall_grace_seconds: float = 45.0
    stall_penalty_per_10s: float = -0.10
    stall_penalty_cap: float = -2.0
    swing_penalty: float = -1.0
    approach_radius: float = 0.05
    pass_displacement: float = 0.008
    position_tolerance: float = 0.002

    def __post_init__(self):
        d = asdict(self)
        if not all(isfinite(x) for x in d.values()):
            raise ValueError('All reward parameters must be finite')
        if not (0 <= self.yield_bonus <= 2):
            raise ValueError('yield_bonus must be in [0,2]')
        if not (1 <= self.stall_grace_seconds <= 1800):
            raise ValueError('stall_grace_seconds out of range')
        if self.stall_penalty_per_10s > 0 or self.stall_penalty_cap > 0 or self.swing_penalty > 0:
            raise ValueError('penalties must be nonpositive')
        if not (0 < self.position_tolerance < self.pass_displacement < self.approach_radius):
            raise ValueError('Invalid distance thresholds')

    def metadata(self):
        return asdict(self)


class SafeYieldStallReward:
    """Episode-scoped states keyed by physical AGV, not changing PettingZoo aliases.

    `physical_agvs` is the CRITIC'S unshuffled observation from Matrix, indexed
    by physical AGV. Coordinates use Tobias's normalized XY in columns 2:4;
    destination XY are columns 4:6. Stop is action 0.
    """
    def __init__(self, config: SafeYieldConfig | None = None):
        self.config = config or SafeYieldConfig()
        self.reset_episode()

    def reset_episode(self):
        self._states: dict[int, dict[str, Any]] = {}
        self.counts = Counter()
        self.total_yield_bonus = 0.0
        self.total_stall_penalty = 0.0
        self.total_swing_penalty = 0.0

    @staticmethod
    def _distance(a, b):
        return hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))

    @staticmethod
    def _point(row, start):
        if len(row) < start + 2:
            return None
        x, y = float(row[start]), float(row[start + 1])
        return (x, y) if isfinite(x) and isfinite(y) else None

    def _new_state(self, xy, t, goal):
        return dict(last_xy=xy, last_goal=goal, last_t=t, dwell_start=t,
                    previous_action=None, previous_agvs=None,
                    yield_paid=False, dwell_stall_total=0.0,
                    positions=deque([xy], maxlen=2))

    def choose(self, physical: int, action: int):
        """Called after a real routing action is serialized for the simulator."""
        physical, action = int(physical), int(action)
        if not 0 <= physical < 20 or not 0 <= action <= 4:
            raise ValueError('Expected 20 AGV / Routing action 0..4')
        self.counts['routing_decisions'] += 1
        self.counts['stop_decisions' if action == 0 else 'move_decisions'] += 1
        st = self._states.get(physical)
        if st is not None:
            st['previous_action'] = action

    def _yield_was_observed(self, physical, xy, old_agvs, new_agvs):
        """Conservative proxy: a nearby physical AGV moved away after our Stop.

        This is NOT proof of an actual safe handoff or a collision-free pass.
        Exactly one bonus is allowed per continuous stationary spell.
        """
        if old_agvs is None or old_agvs.shape != new_agvs.shape:
            return False
        for j in range(min(20, len(new_agvs))):
            if j == physical:
                continue
            old, new = old_agvs[j], new_agvs[j]
            if (len(old) < 6 or len(new) < 6 or old[0] < 0.5 or new[0] < 0.5):
                continue
            a, b = self._point(old,2), self._point(new,2)
            if a is None or b is None:
                continue
            d0, d1 = self._distance(a,xy), self._distance(b,xy)
            # Require proximity BEFORE the Stop, real movement, and clearance.
            if (0.006 < d0 <= self.config.approach_radius
                and self._distance(a,b) >= self.config.pass_displacement
                and d1 >= d0 + self.config.pass_displacement):
                return True
        return False

    def observe(self, physical: int, physical_agvs, model_time: float,
                raw_reward: float, info=None) -> float:
        """Shape the *current Routing reward* BEFORE collector.last() consumes it."""
        physical = int(physical)
        t, reward = float(model_time), float(raw_reward)
        if not isfinite(t) or t < 0 or not isfinite(reward):
            raise ValueError('Nonfinite/negative AnyLogic time or nonfinite reward')
        agvs = np.asarray(physical_agvs, dtype=float)
        self.counts['routing_observations'] += 1
        if agvs.ndim != 2 or len(agvs) <= physical or agvs.shape[1] < 8:
            self.counts['missing_observation'] += 1
            self._states.pop(physical, None)
            return reward
        ego = agvs[physical]
        xy = self._point(ego, 2)
        if ego[0] < 0.5 or xy is None:
            # Avoid credit flowing across an AGV replacement / inactive interval.
            self.counts['inactive_ego'] += 1
            self._states.pop(physical, None)
            return reward
        st = self._states.get(physical)
        if st is None:
            self._states[physical] = self._new_state(xy, t, self._point(ego, 6))
            self._states[physical]['previous_agvs'] = agvs.copy()
            return reward
        if t < st['last_t']-1e-5:
            raise RuntimeError(f'Model time went backward for physical AGV {physical}')

        action = st['previous_action']
        goal = self._point(ego, 6)
        changed_goal = (goal is None) != (st['last_goal'] is None)
        if goal is not None and st['last_goal'] is not None:
            changed_goal = changed_goal or (self._distance(goal, st['last_goal']) > self.config.position_tolerance)
        moved = self._distance(xy, st['last_xy']) > self.config.position_tolerance
        if changed_goal:
            # A new dispatch target may be assigned after a long processing wait.
            # Do NOT charge that processing time as traffic congestion.
            st['dwell_start'] = t
            st['dwell_stall_total'] = 0.0
            st['yield_paid'] = False
            self.counts['target_changes_reset_dwell'] += 1
        if moved:
            self.counts['physical_position_changes'] += 1
            # A -> B -> A penalized ONLY after two prior distinct positions.
            h = st['positions']
            if (action is not None and action != 0 and len(h)==2
                and self._distance(xy,h[0]) <= self.config.position_tolerance
                and self._distance(xy,h[1]) > self.config.position_tolerance):
                reward += self.config.swing_penalty
                self.total_swing_penalty += self.config.swing_penalty
                self.counts['swing_penalties_applied'] += 1
            h.append(xy)
            st['dwell_start'] = t
            st['dwell_stall_total'] = 0.0
            st['yield_paid'] = False
        else:
            at_target = (goal is not None and
                         self._distance(goal, xy) <= self.config.position_tolerance)
            if at_target:
                self.counts['stationary_at_target_exempt'] += 1
                st['dwell_start'] = t
                st['dwell_stall_total'] = 0.0
                st['yield_paid'] = False
            elif not changed_goal:
                old_agvs = st['previous_agvs']
                if action == 0 and not st['yield_paid'] and self._yield_was_observed(
                    physical, xy, old_agvs, agvs):
                    reward += self.config.yield_bonus
                    self.total_yield_bonus += self.config.yield_bonus
                    self.counts['yield_proxy_bonuses_applied'] += 1
                    st['yield_paid'] = True
                # Penalize only INCREMENTAL simulated seconds beyond the grace.
                # No event-count penalty; no double charging at identical timestamps.
                old_age = max(0., st['last_t'] - st['dwell_start'])
                new_age = max(0., t - st['dwell_start'])
                extra_s = (max(0.,new_age-self.config.stall_grace_seconds)
                           - max(0.,old_age-self.config.stall_grace_seconds))
                if extra_s > 0 and action is not None:
                    self.counts['stalled_observations_over_grace'] += 1
                    penalty = self.config.stall_penalty_per_10s * extra_s / 10.
                    remaining = max(0.,-self.config.stall_penalty_cap - st['dwell_stall_total'])
                    penalty = max(penalty, -remaining)
                    if penalty < 0:
                        reward += penalty
                        self.total_stall_penalty += penalty
                        st['dwell_stall_total'] += -penalty
                        self.counts['stall_penalties_applied'] += 1
                        self.counts['stall_penalized_seconds_proxy'] += extra_s
        st['last_xy'] = xy
        st['last_goal'] = goal
        st['last_t'] = t
        st['previous_agvs'] = agvs.copy()
        st['previous_action'] = None
        return reward

    def report(self):
        result = dict(self.counts)
        result.update(mode='direct_environment_safe_yield_stall_v1',
                      configuration=self.config.metadata(),
                      total_yield_bonus=self.total_yield_bonus,
                      total_stall_penalty=self.total_stall_penalty,
                      total_swing_penalty=self.total_swing_penalty,
                      total_reward_adjustment=(self.total_yield_bonus
                        +self.total_stall_penalty+self.total_swing_penalty),
                      note='Yield is clearance proxy, stall uses unchanged-position simulation time; only routing is shaped')
        return result

    def install(self, behaviors_module=None, matrix_module=None):
        """Process-local environment hooks; no permanent changes to source files."""
        if behaviors_module is None:
            from thesis.envs import behaviors_v121 as behaviors_module
        if matrix_module is None:
            from thesis.envs import matrix_v121 as matrix_module
        behavior_cls = behaviors_module.MultiAgent
        matrix_cls = matrix_module.Matrix
        if (getattr(behavior_cls, '_v14_safe_yield_installed', False)
                or getattr(matrix_cls, '_v14_safe_yield_installed', False)):
            raise RuntimeError('Safe-yield hook already installed; do not combine reward modes')
        # Refuse mixed modes even when the old wrapper has already been installed.
        if getattr(behavior_cls, '_v14_reward_wrapper_installed', False):
            raise RuntimeError('Old Stop/Swing reward wrapper is already installed')
        original_action = behavior_cls.convert_to_action
        original_save = matrix_cls._save_observation
        shaping = self

        def safe_action(instance, actions, agent):
            result = original_action(instance, actions, agent)
            if not getattr(instance, 'dispatching', False):
                try:
                    physical = int(agent)
                except (TypeError, ValueError):
                    return result
                if 0 <= physical < 20:
                    raw = actions[0] if isinstance(actions, (list,tuple)) else actions
                    shaping.choose(physical, 0 if raw is None else int(raw))
            return result

        def safe_save(env, alpyne_obs, agent, obs, reward, done, info):
            result = original_save(env, alpyne_obs, agent, obs, reward, done, info)
            caller = int(alpyne_obs.caller)
            if 0 <= caller < 20 and agent != 'dummy':
                # Original Matrix already captured the real model clock and the
                # unshuffled physical-index AGV state for the centralized critic.
                global_state = env.get_mappo_state()
                env.rewards[agent] = shaping.observe(
                    caller, global_state['agvs'], env.get_mappo_time(),
                    env.rewards[agent], info=info)
            return result

        behavior_cls.convert_to_action = safe_action
        matrix_cls._save_observation = safe_save
        behavior_cls._v14_safe_yield_installed = True
        matrix_cls._v14_safe_yield_installed = True
