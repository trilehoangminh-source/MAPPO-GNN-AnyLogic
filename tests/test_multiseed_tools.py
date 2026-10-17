"""Pure-Python contract test: no AnyLogic/torch required."""
import importlib.util
import json
import tempfile
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'scripts'

def load(name):
    spec = importlib.util.spec_from_file_location(name, P / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

runner = load('run_stop_ab_multiseed')
summary = load('summarize_stop_ab_multiseed')


def mock_rows(seed, mode, episodes=2):
    rows = []
    for episode in range(1, episodes + 1):
        b = mode == 'forbid_when_move'
        rows.append({
            'episode': episode,
            'experiment': {'seed': seed, 'stop_mask_mode': mode, 'sim_seconds': 720,
                           'routing_backbone': 'pooled', 'dispatch_backbone': 'pooled'},
            'environment': {
                'delivered_reward': -500.0 + (20 if b else 0) + episode,
                'events': 100 + episode,
                'statistics': {'jobDone': 0, 'targetReached': episode + int(b),
                               'wrongTarget': 20 + (3 if b else 0),
                               'taken_invalid': 0 if b else 20,
                               'block': 5, 'liveLock': 2},
                'action_diagnostics': {'actions_by_role': {'routing': {'0': 0 if b else 30, '1': 70}},
                                       'routing_stop_masked_decisions': 50 if b else 0,
                                       'selected_mask_violations': 0},
                'station_state_sources': {'fresh_routing': 90},
            },
            'policy_updates': {'routing': {'transitions': 90},
                               'dispatch_front': {'transitions': 1},
                               'dispatch_rear': {'transitions': 1}},
        })
    return rows


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    for seed in (44, 45):
        for arm, mode in [('A','baseline'), ('B','forbid_when_move')]:
            output = root / f'seed_{seed}' / arm
            output.mkdir(parents=True)
            metrics = output / 'metrics.jsonl'
            metrics.write_text('\n'.join(json.dumps(r) for r in mock_rows(seed, mode)) + '\n', encoding='utf-8')
            assert runner.validate_completed(output, seed, mode, 2, 720)
            assert not runner.validate_completed(root / 'untouched', seed, mode, 2, 720)
            try:
                runner.validate_completed(output, seed, mode, 3, 720)
                raise AssertionError('Expected mismatch for episodes')
            except RuntimeError:
                pass
    summary.main(['--root', str(root), '--seeds', '44', '45'])
    paired_file = root / 'paired_seed_summary.csv'
    assert paired_file.is_file()
    rows = (root / 'aggregate_summary.csv').read_text(encoding='utf-8-sig')
    assert 'targetReached' in rows and 'wrongTarget' in rows
    assert 'taken_invalid' in rows
    assert 'reward' in rows
print('PASS: paired-seed validation, safeguards, and CSV aggregation')
