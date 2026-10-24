"""Standalone offline checks; does not import torch or start AnyLogic."""
from pathlib import Path
import importlib.util
import logging
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
launcher = root / 'scripts/run_v8_quiet.py'
spec = importlib.util.spec_from_file_location('run_v8_quiet', launcher)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
noisy = "observation.networkcontext = None (<class 'NoneType'>) was not parsed to the expected type (<class 'list'>)"


def rec(level, message, name='alpyne.data'):
    return logging.LogRecord(name, level, __file__, 1, message, (), None)

f = m.AlpyneNetworkContextFilter()
assert f.filter(rec(logging.WARNING, noisy)) is False
assert f.filter(rec(logging.WARNING, 'Another warning')) is True
assert f.filter(rec(logging.ERROR, noisy)) is True
assert f.filter(rec(logging.WARNING, noisy, 'other.library')) is True
m.install_quiet_filter()
m.install_quiet_filter()
assert sum(isinstance(x,m.AlpyneNetworkContextFilter) for x in logging.getLogger('alpyne.data').filters) == 1

# Verify subprocess proxy rewrites ONLY the V8 evaluation command.
class FakeSubprocess:
    def run(self, command, *args, **kwargs):
        return command

proxy = m.QuietV8EvaluatorSubprocess(FakeSubprocess())
eval_path = root/'scripts/evaluate_B3_MAPPO_v7.py'
args = ['python','-u',str(eval_path),'--checkpoint','some.pt']
changed = proxy.run(args)
assert changed[2].endswith('_v8_quiet_evaluate.py')
assert changed[3] == str(eval_path)
assert changed[4:] == ['--checkpoint','some.pt']
assert proxy.run(['python','unrelated.py']) == ['python','unrelated.py']

# Verify our filter does hide only the one known warning on an actual logger.
log = logging.getLogger('alpyne.data')
old_prop, old_level = log.propagate, log.level
log.propagate = False
log.setLevel(logging.WARNING)
class Capture(logging.Handler):
    def __init__(self):
        super().__init__(); self.lines = []
    def emit(self, record):
        self.lines.append(record.getMessage())
handler = Capture()
log.addHandler(handler)
try:
    log.warning(noisy)
    log.warning('Another warning')
    log.error('Other error')
finally:
    log.removeHandler(handler)
    log.propagate, log.level = old_prop, old_level
assert handler.lines == ['Another warning','Other error'],handler.lines

print('V8 QUIET EPISODE CONSOLE OFFLINE TEST: PASS')

# Exercise the evaluator wrapper in a separate Python interpreter.
import tempfile
with tempfile.TemporaryDirectory() as d:
    fake = Path(d)/'evaluate_B3_MAPPO_v7.py'
    fake.write_text(
        "import logging\n"
        "log=logging.getLogger('alpyne.data')\n"
        "log.warning(\"observation.networkcontext = None (<class 'NoneType'>) was not parsed to the expected type (<class 'list'>)\")\n"
        "log.warning('KEEP_OTHER_WARNING')\n"
        "print('EVAL_DID_RUN')\n", encoding='utf-8'
    )
    r=subprocess.run([sys.executable,str(root/'scripts/_v8_quiet_evaluate.py'),str(fake)], capture_output=True,text=True,check=True)
    assert 'observation.networkcontext' not in r.stderr, r.stderr
    assert 'KEEP_OTHER_WARNING' in r.stderr,r.stderr
    assert 'EVAL_DID_RUN' in r.stdout,r.stdout

print('V8 QUIET EVALUATOR SUBPROCESS TEST: PASS')
