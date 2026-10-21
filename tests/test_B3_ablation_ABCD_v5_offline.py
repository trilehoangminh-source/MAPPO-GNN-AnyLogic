"""Standalone stdlib tests: no Torch, no Java/AnyLogic required."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SRC))
from ablation_abcd_v5_common import ARMS, run_dir, train_command, eval_command, assert_distinct_seeds
from analyze_B3_ablation_ABCD_v5 import read_records, calculate, report


class AblationTests(unittest.TestCase):
    def test_factor_grid_and_constant_guide(self):
        self.assertEqual([(a.code, a.encoder, a.guide) for a in ARMS], [
            ('A','gat',1.5), ('B','gat',0.5), ('C','edge_gat',1.5), ('D','edge_gat',0.5)])
        for arm in ARMS:
            out = Path('result') / arm.code
            cmd = train_command('python', Path('project'), Path('model.zip'), arm, 43,
                                2, 720, 20000, 51780, 32, 2, 2, out)
            self.assertEqual(cmd[cmd.index('--routing-guide-strength')+1], str(arm.guide))
            self.assertEqual(cmd[cmd.index('--output')+1], str(out))
            self.assertIn('--min-candidate-map-rate', cmd)
            if arm.encoder == 'gat':
                self.assertEqual(cmd[cmd.index('--routing-encoder')+1], 'gat')
                self.assertNotIn('--guide-final', cmd)
            else:
                self.assertEqual(cmd[cmd.index('--guide-final')+1], str(arm.guide))
                self.assertEqual(cmd[cmd.index('--guide-decay-episodes')+1], '2')
            ev = eval_command('python', Path('project'), arm, Path('checkpoint.pt'),
                              [1001,1002,1003], 3600, 51781, Path('eval.jsonl'))
            self.assertIn('1002', ev)
            self.assertEqual(ev[ev.index('--sim-seconds')+1], '3600')

    def test_seed_leakage(self):
        assert_distinct_seeds([43,143,243], [1001,1002,1003], 2)
        with self.assertRaises(ValueError):
            assert_distinct_seeds([43], [44,1002], 2)
        with self.assertRaises(ValueError):
            assert_distinct_seeds([43], [1001,1001], 2)

    @staticmethod
    def make_fixture(base):
        manifest = {'protocol':'B3_ABCD_V5_factorial_v1', 'train_seeds':[43,143,243],
                    'eval_seeds':[1001,1002,1003], 'train_episodes':2,
                    'train_sim_seconds':720, 'eval_sim_seconds':3600}
        (base/'experiment_manifest.json').write_text(json.dumps(manifest))
        for arm in ARMS:
            for train_seed in manifest['train_seeds']:
                p = run_dir(base,arm,train_seed)
                p.mkdir(parents=True, exist_ok=True)
                cfg = {'seed':train_seed,'episodes':2,'sim_seconds':720,
                       'dispatch_backbone':'pooled','dispatch_mask_mode':'operation_first',
                       'graph_fingerprint':'shared-graph', 'model_file':'model.zip',
                       'max_events':20000,'fleetsize':8,'max_fleetsize':20,
                       'single_dispatch_policy':False,'stop_mask_mode':'forbid_when_move',
                       'graph_layers':2,'hidden':32,'gamma':0.9,'gae_lambda':0.95,
                       'clip':0.1,'lr':0.0001,'epochs':5,'mini_batches':1,
                       'routing_encoder':'gat' if arm.encoder == 'gat' else 'residual_edge_gat'}
                if arm.encoder=='gat':cfg['routing_guide_strength']=arm.guide
                else:cfg.update(guide_start=arm.guide, guide_final=arm.guide)
                (p/'run_config.json').write_text(json.dumps(cfg))
                with (p/'evaluation_3600.jsonl').open('w') as f:
                    for es in manifest['eval_seeds']:
                        # Controlled known factorial effect: baseline 10; Edge +2;
                        # guide 0.5 +3; interaction +4.
                        effect = 2*(arm.encoder=='edge_gat') + 3*(arm.guide==0.5) + 4*(arm.code=='D')
                        job = 10 + effect + (train_seed-43)/100 + (es-1001)*0.1
                        stats = {'jobDone':job,'targetReached':100,'wrongTarget':5,
                                 'block':50-effect,'liveLock':20}
                        row = {'seed':es, 'environment':{
                            'statistics':stats, 'delivered_reward':job*5, 'reached_terminal':True,
                            'dijkstra_strength_eval':arm.guide}}
                        f.write(json.dumps(row)+'\n')
        return manifest

    def test_aggregate_paired_effects_and_strict_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.make_fixture(base)
            cfg, records = read_records(base)
            self.assertEqual(len(records), 36)
            summary, effects = calculate(cfg, records)
            d = {(x['metric'],x['contrast']):x for x in effects}
            self.assertAlmostEqual(d['jobDone','EdgeGAT_vs_GAT']['average'], 4.0)
            self.assertAlmostEqual(d['jobDone','Guide0p5_vs_1p5']['average'], 5.0)
            self.assertAlmostEqual(d['jobDone','Interaction_difference_in_differences']['average'], 4.0)
            self.assertEqual(d['jobDone','EdgeGAT_vs_GAT']['n_training_seeds'], 3)
            path = report(base)
            self.assertTrue(path.is_file())
            self.assertTrue((base/'factorial_contrasts.csv').exists())
            # Strict evaluator refuses missing seed, not silently dropping it
            p = run_dir(base,ARMS[3],43)/'evaluation_3600.jsonl'
            p.write_text('\n'.join(p.read_text().splitlines()[:2])+'\n')
            with self.assertRaises(RuntimeError):
                read_records(base)


if __name__=='__main__':
    unittest.main(verbosity=2)
