"""Offline protocol tests: no AnyLogic, Java, torch, or model ZIP required."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from v6_cd_common import CONDITIONS, seed_validation, condition_dir, training_cmd, evaluation_cmd
from run_B3_V6_CD_replication import manifest, valid_train, valid_eval
from analyze_B3_V6_CD_replication import collect, summarize, make_report


def fake_args():
    return SimpleNamespace(train_seeds=[43, 143, 243], eval_seeds=[2001, 2002, 2003, 2004, 2005],
        train_episodes=30, train_sim_seconds=720, eval_sim_seconds=3600,
        max_events=20000, hidden=32, graph_layers=2, heads=2, epochs=5)


class V6ProtocolTests(unittest.TestCase):
    def test_only_c_and_d_and_constant_guidance(self):
        self.assertEqual([(c.code, c.guide) for c in CONDITIONS], [('C', 1.5), ('D', .5)])
        a = fake_args()
        for c in CONDITIONS:
            d = condition_dir(Path('runs'), c, 43)
            cmd = training_cmd('python', Path('pr'), Path('model.zip'), c, 43, 30,
                               720, 20000, 51830, 32, 2, 2, 5, d)
            self.assertEqual(cmd[cmd.index('--guide-final') + 1], str(c.guide))
            self.assertEqual(cmd[cmd.index('--routing-guide-strength') + 1], str(c.guide))
            self.assertIn('train_B3_edgeGAT_MAPPO_v4.py', cmd[2])
            self.assertNotIn('train_B3_GNN_UOE_v2.py', ' '.join(cmd))
            self.assertEqual(cmd[cmd.index('--epochs') + 1], '5')
            self.assertEqual(cmd[cmd.index('--min-candidate-map-rate') + 1], '0.99')
            ecmd = evaluation_cmd('python', Path('pr'), d / 'checkpoint_ep_0030.pt',
                                  a.eval_seeds, 3600, 51831, d / 'evaluation_3600.jsonl')
            self.assertIn('2005', ecmd)

    def test_eval_seed_leakage_and_duplicate_rejected(self):
        seed_validation([43, 143, 243], [2001, 2002, 2003, 2004, 2005], 50)
        with self.assertRaises(ValueError):
            seed_validation([43], [46], 30)
        with self.assertRaises(ValueError):
            seed_validation([43, 43], [2001], 30)
        with self.assertRaises(ValueError):
            seed_validation([43], [2001, 2001], 30)

    @staticmethod
    def fixture(folder: Path):
        a = fake_args()
        model = Path('C:/dummy-model.zip').resolve()
        mf = manifest(a, model)
        (folder / 'experiment_manifest.json').write_text(json.dumps(mf), encoding='utf-8')
        for c in CONDITIONS:
            for ts in a.train_seeds:
                d = condition_dir(folder, c, ts)
                d.mkdir(parents=True, exist_ok=True)
                ckpt = d / 'checkpoint_ep_0030.pt'
                ckpt.write_bytes(b'fake-ckpt-' * 200)
                config = {
                    'version': 'B3_EDGE_GAT_MAPPO_V4', 'routing_encoder': 'residual_edge_gat',
                    'dispatch_backbone': 'pooled', 'dispatch_mask_mode': 'operation_first',
                    'graph_fingerprint': 'same-graph', 'model_file': str(model),
                    'seed': ts, 'episodes': 30, 'sim_seconds': 720, 'max_events': 20000,
                    'hidden': 32, 'graph_layers': 2, 'heads': 2, 'epochs': 5,
                    'mini_batches': 1, 'stop_mask_mode': 'forbid_when_move',
                    'guide_start': c.guide, 'guide_final': c.guide,
                    'final_used_guide_strength': c.guide,
                    'gamma': .9, 'gae_lambda': .95, 'clip': .1, 'lr': .0001,
                    'fleetsize': 8, 'max_fleetsize': 20, 'single_dispatch_policy': False,
                }
                (d / 'run_config.json').write_text(json.dumps(config))
                with (d / 'metrics.jsonl').open('w') as f:
                    for i in range(30):
                        f.write(json.dumps({'episode': i + 1, 'environment_seed': ts + i}) + '\n')
                with (d / 'evaluation_3600.jsonl').open('w') as f:
                    for es in a.eval_seeds:
                        # Controlled known pairing: D increases job by 2 and lowers block by 4.
                        val = 5 + (ts - 43) / 100 + (es - 2001) / 20 + (2 if c.code == 'D' else 0)
                        stats = {'jobDone': val, 'targetReached': 100 + val,
                                 'wrongTarget': 6, 'block': 50 - (4 if c.code == 'D' else 0),
                                 'liveLock': 20}
                        f.write(json.dumps({'evaluation': True, 'version': 'B3_EDGE_GAT_MAPPO_V4',
                            'seed': es, 'checkpoint': str(ckpt.resolve()),
                            'environment': {'reached_terminal': True,
                                            'delivered_reward': 10 * val,
                                            'dijkstra_strength_eval': c.guide,
                                            'statistics': stats}}) + '\n')
        return a, model

    def test_paired_aggregation_with_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            a, model = self.fixture(folder)
            for c in CONDITIONS:
                for ts in a.train_seeds:
                    d = condition_dir(folder, c, ts)
                    self.assertTrue(valid_train(d, c, ts, a, model))
                    self.assertTrue(valid_eval(d / 'evaluation_3600.jsonl', c, a,
                                               d / 'checkpoint_ep_0030.pt'))
            m, rows = collect(folder)
            self.assertEqual(len(rows), 30)
            seeds, summary = summarize(m, rows)
            self.assertEqual(len(seeds), 3)
            by_metric = {r['metric']: r for r in summary}
            self.assertAlmostEqual(by_metric['jobDone']['D_minus_C'], 2)
            self.assertEqual(by_metric['jobDone']['D_wins_training_seeds'], 3)
            self.assertAlmostEqual(by_metric['block']['D_minus_C'], -4)
            self.assertTrue(make_report(folder).exists())
            self.assertTrue((folder / 'V6_per_training_seed.csv').is_file())
            self.assertTrue((folder / 'V6_CD_REPORT.md').is_file())

    def test_missing_eval_or_incompatible_config_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            a, model = self.fixture(folder)
            d = condition_dir(folder, CONDITIONS[1], 143)
            ev = d / 'evaluation_3600.jsonl'
            ev.write_text('\n'.join(ev.read_text().splitlines()[:-1]) + '\n')
            self.assertFalse(valid_eval(ev, CONDITIONS[1], a, d / 'checkpoint_ep_0030.pt'))
            with self.assertRaises(RuntimeError):
                collect(folder)
            self.fixture(folder)
            conf = d / 'run_config.json'
            data = json.loads(conf.read_text())
            data['guide_final'] = 1.5
            conf.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):
                valid_train(d, CONDITIONS[1], 143, a, model)
            with self.assertRaises(RuntimeError):
                collect(folder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
