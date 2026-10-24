"""Execute the unchanged V7 evaluator while hiding only one Alpyne warning.

Example (normally launched automatically by run_v8_quiet.py):
python _v8_quiet_evaluate.py evaluate_B3_MAPPO_v7.py --checkpoint ...
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

from run_v8_quiet import install_quiet_filter


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Missing path to evaluation script')
    evaluator = Path(sys.argv[1]).resolve()
    if evaluator.name != 'evaluate_B3_MAPPO_v7.py' or not evaluator.is_file():
        raise ValueError('Only evaluate_B3_MAPPO_v7.py is accepted')
    install_quiet_filter()
    sys.argv = [str(evaluator), *sys.argv[2:]]
    runpy.run_path(str(evaluator), run_name='__main__')


if __name__ == '__main__':
    main()
