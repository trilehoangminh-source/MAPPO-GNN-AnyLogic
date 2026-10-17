"""Offline mock: greedy held-out evaluator never updates weights."""
from __future__ import annotations
import sys, json, tempfile, types, importlib.util
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts'
class Env:
    def __init__(self):self._mappo_source_counts=Counter()
    def close(self):pass
class Model:
    trained=False
    def load(self, path): self.path=path
class Collector:
    def __init__(self,model, env,stop_mask_mode): assert stop_mask_mode=='baseline';self.env=env
    def episode(self,seed,train,max_event_steps):
        assert train is False
        self.env._mappo_source_counts['fresh_routing']+=5
        return [],{'delivered_reward':-100+seed/1000,'events':6,
                  'statistics':{'jobDone':0,'targetReached':2,'wrongTarget':1,'taken_invalid':10}}
module=types.ModuleType('train_mappo_stop_ab')
module.build_env=lambda args:Env()
module.make_model=lambda env,args:Model()
sys.modules['train_mappo_stop_ab']=module
for name in ['thesis','thesis.mappo']:
    t=types.ModuleType(name); t.__path__=[];sys.modules[name]=t
cm=types.ModuleType('thesis.mappo.collector_stop_ab');cm.EventCollector=Collector
sys.modules['thesis.mappo.collector_stop_ab']=cm
spec=importlib.util.spec_from_file_location('eval_long',SCRIPTS/'evaluate_mappo_long_v121.py')
script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp)
    ck=d/'checkpoint_ep_0050.pt';ck.write_bytes(b'fake')
    settings={'stop_mask_mode':'baseline','routing_backbone':'pooled','dispatch_backbone':'pooled',
              'seed':43,'episodes':50,'model_file':str(d/'model.zip'),
              'fleetsize':8,'max_fleetsize':20,'max_events':20000,
              'single_dispatch_policy':False,'gamma':.9,'gae_lambda':.95,
              'clip':.1,'lr':1e-4,'epochs':5,'mini_batches':1,'hidden':32}
    (d/'model.zip').write_bytes(b'fake')
    (d/'run_config.json').write_text(json.dumps(settings))
    sys.argv=['eval','--checkpoint',str(ck),'--seeds','105','106','107','--output',str(d/'eval.jsonl')]
    script.main()
    rows=[json.loads(s) for s in (d/'eval.jsonl').read_text().splitlines()]
    assert [r['seed'] for r in rows]==[105,106,107]
    assert all(r['environment']['station_state_sources']['fresh_routing']==5 for r in rows)
    assert ck.read_bytes()==b'fake'
print('PASS: evaluator held-out seeds, greedy no-train, result logging, checkpoint unchanged')
