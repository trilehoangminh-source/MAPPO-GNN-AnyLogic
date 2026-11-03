"""Offline tests for STOP-UNLOCK fork metadata, checkpoint integrity, and masks.

Run: python -m unittest discover -s tests -p 'test_v14_stop_unlocked_offline.py' -v
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'train_v14_stop_unlocked.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_runner():
    stub=types.ModuleType('v14_deep_mlp_common')
    stub.build_v14=lambda *a,**k: None
    spec=importlib.util.spec_from_file_location('v14_stop_unlocked_for_test',SCRIPT)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'v14_deep_mlp_common':stub}):
        spec.loader.exec_module(module)
    return module


class StopUnlockedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=load_runner()

    def source_fixture(self, root, ep=92):
        root=Path(root)
        source=root/'previous_v14'
        source.mkdir()
        model_file=root/'model.zip'
        model_file.write_bytes(b'example AnyLogic ZIP placeholder for checksum tests')
        cfg={'model_file':str(model_file),'graph_fingerprint':'example-sha',
             'stop_mask_mode':'forbid_when_move','ppo_v7_options':{'value_loss':'clipped_huber'}}
        blueprint=source/'run_config_blueprint.json'
        blueprint.write_text(json.dumps(cfg))
        cp=source/f'rolling_slot_{ep%2}.pt'
        cp.write_bytes(b'example original trained checkpoint, do not overwrite')
        meta={'version':self.r.SOURCE_VERSION,'stop_mask_mode':'forbid_when_move',
              'source_blueprint_sha256':sha(blueprint), 'anylogic_sha256':sha(model_file),
              'fleetsize':20,'gat_layers':4,'seed':43,'max_events':30000,
              'sim_seconds':3600,'max_minibatch':128,'gat_heads':2,'dispatch_width':128,
              'embedding':64,'station_id_dim':16,'cpu_threads':2,'snapshot_every':10,
              'device':'cuda:0','port':51880}
        (source/'experiment.json').write_text(json.dumps(meta))
        (source/'state.json').write_text(json.dumps({'last_episode':ep,'checkpoint':str(cp)}))
        (source/'metrics.jsonl').write_text(''.join(json.dumps({'episode':x})+'\n' for x in range(1,ep+1)))
        return source,cp

    def test_mode_is_baseline(self):
        self.assertEqual(self.r.STOP_MODE,'baseline')

    def test_parse_fork_args(self):
        args=self.r.parse_args(['--fork-from','source','--max-minibatch','128','--init-only'])
        self.assertEqual(args.max_minibatch,128)
        self.assertTrue(args.init_only)
        self.assertFalse(args.resume)

    def test_dry_run_of_committed_source_without_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            source,cp=self.source_fixture(folder)
            dest=Path(folder)/'new_run'
            self.r.main(['--fork-from',str(source),'--output',str(dest),
                         '--max-minibatch','128','--max-episodes','1100','--dry-run'])
            self.assertFalse(dest.exists())
            self.assertEqual(cp.read_bytes(),b'example original trained checkpoint, do not overwrite')

    def test_refuses_wrong_batch_size_on_fork(self):
        with tempfile.TemporaryDirectory() as folder:
            source,_=self.source_fixture(folder)
            with self.assertRaisesRegex(RuntimeError,'Fork args differ'):
                self.r.main(['--fork-from',str(source),'--output',str(Path(folder)/'new_run'),
                             '--max-minibatch','16','--dry-run'])

    def test_fork_resume_metrics_start_after_parent_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'metrics.jsonl'
            p.write_text(json.dumps({'episode':93})+'\n'+json.dumps({'episode':94})+'\n')
            self.r.metrics_preflight(p,94,92)
            with self.assertRaisesRegex(RuntimeError,'Missing/duplicated'):
                self.r.metrics_preflight(p,95,92)

    def test_old_metrics_not_modified_on_source_validation_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            source,_=self.source_fixture(folder)
            p=source/'metrics.jsonl'
            original=p.read_bytes()
            p.write_bytes(original.replace(b'"episode": 35',b'"episode": 3500'))
            corrupt=p.read_bytes()
            with self.assertRaisesRegex(RuntimeError,'source metrics missing/duplicated'):
                self.r.main(['--fork-from',str(source),'--output',str(Path(folder)/'new_run'), '--dry-run'])
            self.assertEqual(p.read_bytes(),corrupt)

    def test_baseline_collector_preserves_stop_when_move_exists(self):
        try:
            from thesis.mappo.collector_stop_ab import EventCollector
            import numpy as np
        except ImportError as exc:
            self.skipTest('Original MAPPO source not installed in isolated ZIP extraction: '+str(exc))
        model=types.SimpleNamespace(roles={'routing':types.SimpleNamespace(actor=types.SimpleNamespace(action_dim=5))})
        obs={'agvs':np.zeros((20,20),dtype='float32'), 'stat':np.zeros((27,20),dtype='float32'),
             'action_mask':np.array([1.,1.,1.,0.,0.],dtype='float32')}
        base=EventCollector(model,None,stop_mask_mode='baseline')._prepare(obs,'routing')
        old=EventCollector(model,None,stop_mask_mode='forbid_when_move')._prepare(obs,'routing')
        self.assertEqual(base['action_mask'].tolist(),[1,1,1,0,0])
        self.assertEqual(old['action_mask'].tolist(),[0,1,1,0,0])
        self.assertEqual(obs['action_mask'].tolist(),[1,1,1,0,0])

if __name__=='__main__': unittest.main()
