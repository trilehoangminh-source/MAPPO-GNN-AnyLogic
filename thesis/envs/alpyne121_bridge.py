"""Alpyne >=1.2 compatibility adapter for Tobias's legacy PettingZoo environment.

New code is deliberately isolated: it does not import or monkeypatch Alpyne 0.2.
Only the *_v121.py env modules use this adapter.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from alpyne.sim import AnyLogicSim
from alpyne.constants import EngineState


class Configuration:
    """Legacy Configuration interface expected by Tobias's build_config()."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def names(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())

    def as_dict(self):
        return dict(self.__dict__)


class Observation:
    """Legacy observation facade around new dict-like SimObservation."""
    def __init__(self, data):
        self._values = dict(data)

    def names(self):
        return list(self._values.keys())

    def get(self, key, default=None):
        return self._values.get(key, default)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name):
        return self._values[name]


class Action:
    """Accept the legacy Action(data=[{name,type,value,unit},...]) layout."""
    def __init__(self, data=None, **kwargs):
        if data is not None:
            kwargs.update({entry['name']: entry['value'] for entry in data})
        self.fields = kwargs


class ModelRun:
    """Minimal old ModelRun methods required by BaseAlpyneZoo."""
    def __init__(self, sim: AnyLogicSim, config: Configuration):
        self._sim = sim
        self._config = config
        self._status = None
        self.id = None

    @staticmethod
    def _as_kwargs(config):
        if isinstance(config, Configuration):
            return config.as_dict()
        if isinstance(config, dict):
            return dict(config)
        return vars(config).copy()

    def run(self):
        if self.id is not None:
            raise RuntimeError('ModelRun already started')
        # Old BaseAlpyneZoo._start expects run() to initialize an episode.
        # BaseAlpyneZoo.reset will reset once more; this is safe but adds a reset.
        self._status = self._sim.reset(**self._as_kwargs(self._config))
        self.id = 1
        self._assert_status()
        return self

    def _assert_status(self):
        if self._status is not None and self._status.state == EngineState.ERROR:
            raise RuntimeError(f'AnyLogic engine MODEL_ERROR: {self._status}')

    def wait_for_completion(self):
        if self._status is None:
            self._status = self._sim.lock()
        self._assert_status()
        return self._status

    def reset(self, config=None):
        if config is not None:
            self._config = config
        self._status = self._sim.reset(**self._as_kwargs(self._config))
        self._assert_status()
        return self._status

    def get_observation(self):
        status = self.wait_for_completion()
        if status.observation is None:
            raise RuntimeError(f'AnyLogic returned no observation; engine={status.state}')
        return Observation(status.observation)

    def take_action(self, action):
        if isinstance(action, Action):
            fields = action.fields
        elif isinstance(action, dict):
            fields = action
        else:
            raise TypeError(f'Unsupported action object: {type(action)}')
        # Avoid np.int64/np.ndarray serialization issues for INTEGER_ARRAY.
        receiver = int(fields['receiver'])
        actions = [int(value) for value in fields['actions']]
        self._status = self._sim.take_action(receiver=receiver, actions=actions)
        self._assert_status()
        return self._status

    def is_terminal(self):
        status = self.wait_for_completion()
        return bool(status.stop or status.state == EngineState.FINISHED)

    def get_state(self):
        s = self.wait_for_completion()
        return (s.state, {'model_time': float(s.time)})

    def stop(self):
        # The new Alpyne client manages Java's lifetime. Prefer public cleanup
        # methods when present; do not terminate other processes globally.
        for method in ('close', 'stop'):
            fn = getattr(self._sim, method, None)
            if callable(fn):
                fn()
                return


class AlpyneClient:
    """New AnyLogicSim under a legacy client-facing method name."""
    def __init__(self, model_loc, blocking=False, port=51150, verbose=False):
        java_exe = os.environ.get('MAPPO_JAVA_EXE')
        if not java_exe:
            default = Path(r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
            if default.is_file():
                java_exe = str(default)
        kwargs: dict[str, Any] = dict(
            port=int(port),
            auto_lock=True,
            auto_finish=False,
            max_server_await_time=int(os.environ.get('MAPPO_SERVER_TIMEOUT', '180')),
            lock_defaults={'timeout': int(os.environ.get('MAPPO_LOCK_TIMEOUT', '180'))},
        )
        if java_exe:
            kwargs['java_exe'] = java_exe
        self._sim = AnyLogicSim(str(model_loc), **kwargs)

    def create_reinforcement_learning(self, config):
        return ModelRun(self._sim, config)
