"""Offline parser and device-plumbing test; not a real CUDA/AnyLogic test."""
import argparse
import importlib.util
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
# Stub only V7 args_parser, no JVM / actual checkpoint loaded.
helper=types.ModuleType('_mappo_v8_utils')
for item in ('atomic_json','average_report','read_jsonl','check_resume_state','repair_metrics_on_resume'):
    setattr(helper,item,lambda *a,**kw:None)
sys.modules['_mappo_v8_utils']=helper
class CaptureParser:
    def parse_args(self,argv):
        d={argv[i][2:]:argv[i+1] for i in range(0,len(argv),2)}
        return types.SimpleNamespace(device=d['device'],fleetsize=d['fleetsize'])
v7=types.ModuleType('train_B3_MAPPO_v7');v7.args_parser=CaptureParser
sys.modules['train_B3_MAPPO_v7']=v7
spec=importlib.util.spec_from_file_location('v8_gpu',ROOT/'scripts/train_B3_MAPPO_v8_overnight.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=types.SimpleNamespace(model='fake.zip',port_train=51880,max_additional_episodes=20,train_sim_seconds=1800,max_events=20000,fleetsize=20,device='cuda:0',gamma=None)
config=dict(max_fleetsize=20,seed=43,hidden=32,graph_layers=2,heads=2,gamma=.9,gae_lambda=.95,clip=.1,lr=1e-4,epochs=5,mini_batches=1,ppo_v7_options=dict(value_loss='clipped_huber',value_clip_epsilon=.2,huber_delta=1.,target_kl=.03,time_unit_seconds=0))
module, parsed=m.train_args(a,config)
assert parsed.device=='cuda:0',parsed.device
assert parsed.fleetsize=='20'
assert '--device' in (ROOT/'scripts/train_B3_MAPPO_v8_overnight.py').read_text()
print('V8 GPU DEVICE PLUMBING OFFLINE: PASS (does not prove CUDA or JVM compatibility)')
