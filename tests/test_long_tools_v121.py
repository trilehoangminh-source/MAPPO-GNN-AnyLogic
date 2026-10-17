"""Offline checks for long A trainer outputs and learning curves (no AnyLogic)."""
from __future__ import annotations
import importlib.util
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
from collections import Counter

ROOT=pathlib.Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts'

# Only fake package imports of original MAPPO to test trainer's orchestration.
thesis=types.ModuleType('thesis'); thesis.__path__=[]
mappo=types.ModuleType('thesis.mappo'); mappo.__path__=[]
mappo.MAPPO=type('MAPPO',(),{})
mappo.MAPPOConfig=type('MAPPOConfig',(),{})
coll=types.ModuleType('thesis.mappo.collector_stop_ab')
sys.modules.update({'thesis':thesis,'thesis.mappo':mappo,'thesis.mappo.collector_stop_ab':coll})

class DummyEnv:
    def __init__(self): self._mappo_source_counts=Counter(); self.statistics={}; self.closed=False
    def close(self): self.closed=True

class DummyModel:
    def __init__(self): self.n=0
    def train(self,pieces): self.n+=1;return {'routing': {'transitions':4,'approx_kl':.001}}
    def save(self,path): pathlib.Path(path).write_text(f'n={self.n}', encoding='utf-8')
    def load(self,path): self.n=int(pathlib.Path(path).read_text(encoding='utf-8').split('=')[1])

class DummyCollector:
    calls=0
    fail_at=None
    def __init__(self,model,env,stop_mask_mode='baseline'):
        self.model=model;self.env=env
        assert stop_mask_mode=='baseline'
    def episode(self,seed,train,max_event_steps):
        DummyCollector.calls+=1
        if DummyCollector.calls==DummyCollector.fail_at: raise RuntimeError('Simulated temporary Java failure')
        self.env._mappo_source_counts['fresh_routing']+=4
        return [[{'role':'routing'}]], {'events':4,'closed_transitions':4,'delivered_reward':float(seed),
                'statistics':{'jobDone':0,'targetReached':1,'wrongTarget':2,'taken_invalid':3,'_jobnum_test':1},
                'action_diagnostics':{'actions_by_role':{'routing':{'0':1,'1':3}}}}

coll.EventCollector=DummyCollector
spec=importlib.util.spec_from_file_location('train_mappo_long_v121',SCRIPTS/'train_mappo_long_v121.py')
trainer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
trainer.build_env=lambda args:DummyEnv()
trainer.make_model=lambda env,args:DummyModel()

spec2=importlib.util.spec_from_file_location('plot_mappo_learning_v121',SCRIPTS/'plot_mappo_learning_v121.py')
plot=importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(plot)

with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d)
    model=p/'model.zip';model.write_bytes(b'mock')
    out=p/'run'
    command=['prog','--model',str(model),'--output',str(out),'--episodes','3','--sim-seconds','720','--seed','43']
    sys.argv=command
    DummyCollector.calls=0;DummyCollector.fail_at=2
    try:trainer.main()
    except RuntimeError as ex: assert 'temporary Java' in str(ex)
    else:raise AssertionError('Expected mock failure')
    lines=(out/'metrics.jsonl').read_text().splitlines()
    assert len(lines)==1 and (out/'checkpoint_ep_0001.pt').read_text()=='n=1'
    sys.argv=command+['--resume-existing']
    DummyCollector.fail_at=None
    trainer.main()
    lines=(out/'metrics.jsonl').read_text().splitlines()
    assert len(lines)==3
    assert [json.loads(x)['episode'] for x in lines]==[1,2,3]
    assert [json.loads(x)['environment_seed'] for x in lines]==[43,44,45]
    assert (out/'checkpoint_ep_0003.pt').read_text()=='n=3'
    assert (out/'latest_mappo.pt').read_text()=='n=3'
    rows=plot.summarize_metrics(out/'metrics.jsonl', window=2)
    assert len(rows)==3
    assert [r['routing_action0_pct'] for r in rows]==[25.0]*3
    assert rows[2]['operations']==1
    path,_=plot.save_report(rows,out, window=2, make_plots=False)
    assert path.is_file() and (out/'learning_summary.txt').is_file()
    # Completed resume must not create a new episode.
    trainer.main()
    assert len((out/'metrics.jsonl').read_text().splitlines())==3
    # A fresh (non-resume) command MUST refuse to overwrite.
    sys.argv=command
    with contextlib.redirect_stderr(io.StringIO()):
        try:trainer.main()
        except SystemExit as ex:assert ex.code==2
        else:raise AssertionError('Expected refusal to overwrite')

    # Server may fail even before the FIRST episode; manifest exists, but no
    # metrics yet. Resume should permit the first episode to be retried.
    early=p/'early_failure'
    sys.argv=['prog','--model',str(model),'--output',str(early),'--episodes','1','--sim-seconds','720']
    normal_build=trainer.build_env
    def broken_build(args): raise RuntimeError('Simulated HTTP500 before reset')
    trainer.build_env=broken_build
    try:trainer.main()
    except RuntimeError as ex:assert 'HTTP500' in str(ex)
    else:raise AssertionError('Expected startup failure')
    trainer.build_env=normal_build
    assert (early/'run_config.json').is_file() and not (early/'metrics.jsonl').exists()
    sys.argv.append('--resume-existing')
    trainer.main()
    assert (early/'checkpoint_ep_0001.pt').is_file()
print('PASS: crash-safe long training resume, checkpoints, metrics, learning aggregation')
