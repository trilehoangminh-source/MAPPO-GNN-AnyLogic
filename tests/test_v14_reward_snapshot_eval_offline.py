"""Stdlib-only tests for the V14 10-seed snapshot evaluator (no AnyLogic)."""
import importlib.util
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parents[1]
TARGET = HERE / 'scripts' / 'evaluate_v14_reward_snapshots.py'
spec = importlib.util.spec_from_file_location('v14_snapshot_eval', TARGET)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class TestV14SnapshotEvaluator(unittest.TestCase):
    def test_defaults_exactly_30(self):
        a = e.parse_args([])
        self.assertEqual(a.episodes, [110, 120, 130])
        self.assertEqual(a.seeds, list(range(900001, 900011)))
        self.assertEqual(len(a.episodes)*len(a.seeds), 30)
        self.assertEqual(a.port, 51881)

    def test_reject_duplicate_seeds(self):
        with self.assertRaises(SystemExit):
            e.parse_args(['--seeds', '2', '2'])

    def test_parse_stop_blocktime(self):
        data = {'statistics':{'jobDone':13, 'block':53, 'liveLock':202,
                              '_blocksec_agv1':120.5, '_blocksec_agv2':3,
                              'targetReached':390},
                'action_diagnostics':{'actions_by_role':{'routing':{'0':2,'1':3,'3':5}},
                                      'routing_stop_with_move_available':1},
                'reward_shaping':{'stop_penalties_applied':1,'swing_penalties_applied':2,
                                  'total_reward_adjustment':-2.5},
                'events':10,'end_reason':'environment_terminal','reached_terminal':True,
                'delivered_reward':200.1}
        r = e.parse_result_row(110,900001,data,3600.0,3597.0)
        self.assertEqual(r['jobDone'],13)
        self.assertEqual(r['block_time_s'],123.5)
        self.assertEqual(r['stop_ratio_pct'],20)
        self.assertEqual(r['stop_count'],2)
        self.assertEqual(r['model_time_s'],3600.0)
        self.assertEqual(r['last_decision_time_s'],3597.0)

    def test_paired_differences_per_seed(self):
        rows=[]
        for i,s in enumerate([900001,900002]):
            for ep in [110,120,130]:
                rows.append({'episode':ep,'seed':s,**{k:float(i+ep) for k in e.METRICS}})
        summary,paired=e.rows_summary(rows,[110,120,130],[900001,900002])
        self.assertEqual([x['n_seeds'] for x in summary],[2,2,2])
        x=next(x for x in paired if x['checkpoint_comparison']=='EP130 minus EP110' and x['metric']=='jobDone')
        self.assertEqual(x['mean_delta'],20.0)
        self.assertEqual(x['paired_seeds'],2)
        self.assertIsNone(x['approx_95ci_low'])

    def test_resume_integrity_check(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'heldout.jsonl'
            row=dict(episode=110,seed=900001,checkpoint_sha256='abc',eval_protocol_sha256='proto',
                     reward_implementation_sha256='reward',eval_version=e.VERSION)
            p.write_text(json.dumps(row)+'\n',encoding='utf-8')
            got=e.load_existing(p,[110,120,130],[900001],{110:'abc',120:'x',130:'y'},'proto','reward')
            self.assertIn((110,900001),got)
            with self.assertRaises(RuntimeError):
                e.load_existing(p,[110,120,130],[900001],{110:'DIFFERENT'},'proto','reward')
            p.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n',encoding='utf-8')
            with self.assertRaises(RuntimeError):
                e.load_existing(p,[110,120,130],[900001],{110:'abc'},'proto','reward')

    def test_preflight_dummy_source(self):
        with tempfile.TemporaryDirectory() as d:
            d=pathlib.Path(d); run=d/'mappo_runs'/'test';run.mkdir(parents=True)
            model=d/'anylogic.zip';model.write_bytes(b'anylogic model bytes')
            cfg={'model_file':str(model),'fleetsize':20,'max_fleetsize':20}
            (run/'run_config_blueprint.json').write_text(json.dumps(cfg),encoding='utf-8')
            meta=dict(version=e.SOURCE_VERSION,fleetsize=20,max_fleetsize=20,sim_seconds=3600,
                      gat_layers=4,stop_mask_mode='baseline',reward_mode='direct_behavior_reward',
                      reward_config={'stop_penalty':-0.5},
                      reward_implementation_sha256=e.sha256(e.REWARD_MODULE),
                      source_blueprint_sha256=e.sha256(run/'run_config_blueprint.json'),
                      anylogic_sha256=e.sha256(model),max_events=30000)
            (run/'experiment.json').write_text(json.dumps(meta),encoding='utf-8')
            (run/'state.json').write_text(json.dumps({'last_episode':134}),encoding='utf-8')
            for ep in (110,120,130):
                (run/f'snapshot_ep_{ep:04d}.pt').write_bytes(f'fake-{ep}'.encode())
            _,got,_,cps,hashes,_=e.preflight(run,[110,120,130],[900001])
            self.assertEqual(got['sim_seconds'],3600)
            self.assertEqual(len(cps),3)
            self.assertEqual(len(hashes[110]),64)
            self.assertEqual(len(list(run.iterdir())),6)
            (run/'snapshot_ep_0120.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                e.preflight(run,[110,120,130],[900001])

    def test_file_names_no_training_collision(self):
        a,b,c,d=e.output_paths(pathlib.Path('/x'),[110,120,130],e.DEFAULT_SEEDS)
        self.assertTrue(str(a).endswith('heldout_reward_ep110-120-130_10seeds_3600s.jsonl'))
        self.assertNotIn('metrics.jsonl',str(a))
        self.assertNotIn('state.json',str(a))
        self.assertEqual(len({a,b,c,d}),4)


if __name__=='__main__':
    unittest.main(verbosity=2)
