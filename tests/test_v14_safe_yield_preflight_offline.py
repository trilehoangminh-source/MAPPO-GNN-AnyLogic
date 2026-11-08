"""Read-only source preflight and immutable snapshot checks without AnyLogic."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))


def h(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trainer():
    # The installed full thesis already supplies these modules. Mock imports
    # ONLY inside the offline test so the fork preflight can be exercised.
    shim=types.ModuleType('train_v14_stop_unlocked')
    shim.read_json=lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    shim.atomic_json=lambda path,obj: Path(path).write_text(json.dumps(obj),encoding='utf-8')
    shim.sha256=h
    shim.check_state=lambda st,out: (int(st['last_episode']),Path(st['checkpoint']))
    common=types.ModuleType('v14_deep_mlp_common')
    common.build_v14=lambda *a,**k: None
    path=ROOT/'scripts'/'train_v14_safe_yield_stall.py'
    spec=importlib.util.spec_from_file_location('safe_yield_trainer_to_test',path)
    m=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'train_v14_stop_unlocked':shim,'v14_deep_mlp_common':common}):
        spec.loader.exec_module(m)
    return m


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod=load_trainer()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'old_run';self.source.mkdir()
        self.out=self.root/'new_run'
        self.model=self.root/'model.zip';self.model.write_bytes(b'ANYLOGIC_TEST_MODEL')
        self.old_reward=self.root/'v14_reward_shaping.py';self.old_reward.write_text('test original reward')
        self.cfg={'model_file':str(self.model),'fleetsize':20,'max_fleetsize':20}
        cfgpath=self.source/'run_config_blueprint.json';cfgpath.write_text(json.dumps(self.cfg))
        cp=self.source/'snapshot_ep_0130.pt';cp.write_bytes(b'FAKE_EP130_TENSORS')
        self.source_hash=h(cp)
        self.meta={
            'version':self.mod.SOURCE_VERSION,'reward_mode':'direct_behavior_reward',
            'stop_mask_mode':'baseline','fleetsize':20,'gat_layers':4,
            'sim_seconds':3600,'reward_implementation_sha256':h(self.old_reward),
            'reward_config':{'stop_penalty':-.5},'source_blueprint_sha256':h(cfgpath),
            'anylogic_sha256':h(self.model),'seed':37,'max_events':50000,
            'max_minibatch':128,'gat_heads':2,'dispatch_width':128,'embedding':64,
            'station_id_dim':16,'cpu_threads':2,'snapshot_every':10,'device':'cuda:0','port':51880
        }
        (self.source/'experiment.json').write_text(json.dumps(self.meta))
        (self.source/'state.json').write_text(json.dumps({'last_episode':134}))
        (self.source/'metrics.jsonl').write_text(json.dumps({'episode':130})+'\n')

    def arguments(self):
        return self.mod.parse_args(['--fork-from',str(self.source),
                                    '--output',str(self.out),'--dry-run'])

    def test_exact_snapshot_preflight_is_read_only(self):
        with patch.object(self.mod,'SOURCE_HASH',self.source_hash), patch.object(self.mod,'OLD_REWARD_MODULE',self.old_reward):
            args=self.arguments()
            old_bytes=(self.source/'metrics.jsonl').read_bytes()
            cfg,cfg_path,meta,state,episode,cp,src=self.mod.run_preflight(args)
            self.assertEqual(episode,130)
            self.assertEqual(cp.name,'snapshot_ep_0130.pt')
            self.assertEqual(meta['reward_mode'],'direct_environment_safe_yield_stall')
            self.assertEqual(meta['train_started_at_episode'],131)
            self.assertFalse(self.out.exists())
            self.assertEqual((self.source/'metrics.jsonl').read_bytes(),old_bytes)

    def test_different_checkpoint_rejected(self):
        with patch.object(self.mod,'OLD_REWARD_MODULE',self.old_reward):
            with self.assertRaisesRegex(RuntimeError,'hash differs'):
                self.mod.run_preflight(self.arguments())

    def test_existing_output_never_overwritten(self):
        self.out.mkdir();(self.out/'state.json').write_text('DO NOT DELETE')
        with patch.object(self.mod,'SOURCE_HASH',self.source_hash),patch.object(self.mod,'OLD_REWARD_MODULE',self.old_reward):
            with self.assertRaises(FileExistsError):
                self.mod.run_preflight(self.arguments())
        self.assertEqual((self.out/'state.json').read_text(),'DO NOT DELETE')

    def test_resume_cannot_pass_different_coefficients(self):
        with self.assertRaises(ValueError):
            self.mod.parse_args(['--yield-bonus','-1'])

if __name__=='__main__':unittest.main()
