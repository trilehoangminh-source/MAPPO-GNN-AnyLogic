"""Run from repo root: python tests/test_v8_transfer_resume_metrics.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _mappo_v8_utils import repair_metrics_on_resume

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    out = root / 'from15to20'
    out.mkdir()
    metrics = out / 'metrics.jsonl'
    (out / 'overnight_config.json').write_text(json.dumps({'first_completed_episode': 144}), encoding='utf-8')

    # Crash before first new episode: entirely valid to have no metrics file.
    assert repair_metrics_on_resume(metrics, 144) == 0

    # A tail from a partially committed episode is discarded; no pretend commit.
    metrics.write_text(json.dumps({'episode':145}) + '\n', encoding='utf-8')
    assert repair_metrics_on_resume(metrics, 144) == 0
    assert metrics.read_text(encoding='utf-8') == ''

    # First fully committed 20-AGV episode can be resumed.
    metrics.write_text(json.dumps({'episode':145}) + '\n', encoding='utf-8')
    assert repair_metrics_on_resume(metrics, 145) == 1

    # Missing metrics for a supposedly committed checkpoint must still fail.
    try:
        repair_metrics_on_resume(metrics, 146)
    except RuntimeError as exc:
        assert 'Missing committed metrics' in str(exc)
    else:
        raise AssertionError('Missing committed metric was not detected')

    # 8-AGV V8 starting from episode 30 retains normal behavior.
    origin = root / 'fromv7'
    origin.mkdir()
    assert repair_metrics_on_resume(origin / 'metrics.jsonl', 30) == 0

print('V8 TRANSFER RESUME METRICS FIX: PASS')
