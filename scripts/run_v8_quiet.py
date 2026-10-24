"""Launch the existing V8 trainer and mute one noisy Alpyne schema warning.

All existing MAPPO options work unchanged. The trainer already prints an
episode summary AFTER writing its checkpoint + metrics + state.json.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'


class AlpyneNetworkContextFilter(logging.Filter):
    """Discard ONLY the known None-vs-list Alpyne context warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != 'alpyne.data' or record.levelno != logging.WARNING:
            return True
        msg = record.getMessage()
        return not (
            'observation.networkcontext = None' in msg
            and 'was not parsed to the expected type' in msg
            and "<class 'list'>" in msg
        )


def install_quiet_filter() -> None:
    logger = logging.getLogger('alpyne.data')
    if not any(isinstance(f, AlpyneNetworkContextFilter) for f in logger.filters):
        logger.addFilter(AlpyneNetworkContextFilter())


class QuietV8EvaluatorSubprocess:
    """Proxy used only by this trainer to filter evaluation warnings too."""

    def __init__(self, original_subprocess):
        self._subprocess = original_subprocess

    def run(self, command, *args, **kwargs):
        if isinstance(command, (list, tuple)):
            elements = [str(x) for x in command]
            for i, value in enumerate(elements):
                if Path(value).name == 'evaluate_B3_MAPPO_v7.py':
                    wrapper = SCRIPTS / '_v8_quiet_evaluate.py'
                    if not wrapper.is_file():
                        raise FileNotFoundError(wrapper)
                    command = [*command[:i], str(wrapper), str(command[i]), *command[i+1:]]
                    break
        return self._subprocess.run(command, *args, **kwargs)

    def __getattr__(self, key):
        return getattr(self._subprocess, key)


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not (SCRIPTS / 'train_B3_MAPPO_v8_overnight.py').is_file():
        raise FileNotFoundError('Expected scripts/train_B3_MAPPO_v8_overnight.py in project')
    install_quiet_filter()
    sys.path.insert(0, str(SCRIPTS))
    import train_B3_MAPPO_v8_overnight as trainer
    # Redirect ONLY V8 evaluator commands; no changes to checkpoints or PPO.
    trainer.subprocess = QuietV8EvaluatorSubprocess(trainer.subprocess)
    print('[V8 QUIET] Alpyne networkcontext=None warnings hidden; all other warnings/errors retained.', flush=True)
    print('[V8 QUIET] The original trainer reports each successfully committed episode.', flush=True)
    trainer.main(argv)


if __name__ == '__main__':
    main()
