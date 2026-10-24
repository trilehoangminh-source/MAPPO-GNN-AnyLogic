"""Offline-only sanity checks for 8->15 AGV change."""
import ast
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
ROOT=Path(__file__).resolve().parents[1]
src=(ROOT/'scripts/train_B3_MAPPO_v8_overnight.py').read_text(encoding='utf-8')
ast.parse(src)
sys.path.insert(0,str(ROOT/'scripts'))
import train_B3_MAPPO_v8_overnight as v8
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp)
    ck=d/'cp.pt'; ck.write_bytes(b'fake-checkpoint')
    m=d/'model.zip'; m.write_bytes(b'fake-zip')
    c=d/'run_config.json'
    c.write_text(json.dumps(dict(version='B3_MAPPO_STABILITY_V7',routing_encoder='residual_edge_gat',
       ppo_v7_mode='stable',guide_start=1.5,guide_final=1.5, graph_nodes=105,graph_edges=328,
       model_file=str(m),fleetsize=8,max_fleetsize=20)),encoding='utf-8')
    a=v8.options(['--from-checkpoint',str(ck),'--from-config',str(c),'--fleetsize','15','--dry-run'])
    cfg=v8.load_source(a)
    assert a.fleetsize==15 and cfg['fleetsize']==8
    assert a.source_last_episode is None
    prev=d/'old_8_agv'; prev.mkdir()
    cp_prev=prev/'rolling_slot_1.pt'; cp_prev.write_bytes(b'fake-checkpoint-31')
    (prev/'state.json').write_text(json.dumps(dict(last_episode=31,checkpoint=str(cp_prev.resolve()))),encoding='utf-8')
    (prev/'overnight_config.json').write_text(json.dumps(dict(fleetsize=8)),encoding='utf-8')
    b=v8.options(['--from-previous-run',str(prev),'--from-config',str(c),'--fleetsize','15','--dry-run'])
    cfg2=v8.load_source(b)
    assert b.from_checkpoint==cp_prev.resolve() and b.source_last_episode==31 and b.fleetsize==15
    try:
       v8.load_source(v8.options(['--from-checkpoint',str(ck),'--from-config',str(c),'--fleetsize','21','--dry-run']))
       raise AssertionError('excessive fleet should reject')
    except ValueError: pass
    assert 'staged_baseline' in src and 'effective_config' in src
    assert "'--fleetsize', str(a.fleetsize)" in src
    assert "eval_config['fleetsize'] = a.fleetsize" in src
print('V8 15-AGV OFFLINE PATCH: PASS')
