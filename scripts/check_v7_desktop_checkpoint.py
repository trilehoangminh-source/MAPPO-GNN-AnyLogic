"""Offline checkpoint and compatibility check before using AnyLogic Desktop."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from thesis.pypeline_connectors.v7_inference import verify_checkpoint

p=ROOT/'mappo_runs'/'V7_brain30_seed43'
conf,ckpt=verify_checkpoint(p/'checkpoint_ep_0030.pt',p/'run_config.json')
print('CHECKPOINT: PASS')
print('version:',conf['version'])
print('graph:',conf['graph_nodes'],'nodes,',conf['graph_edges'],'directed edges')
print('routing encoder:',conf['routing_encoder'])
print('dispatch:',conf['dispatch_backbone'])
print('roles:',', '.join(sorted(ckpt['roles'])))
print('guide strength:',conf['final_used_guide_strength'])
print('NOTE: Does not prove AnyLogic Desktop/Pypeline runtime parity.')
