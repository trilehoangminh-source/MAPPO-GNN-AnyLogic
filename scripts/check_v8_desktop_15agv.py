"""Read-only check: V8 selected policy is ready for the existing AnyLogic Pypeline."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from thesis.pypeline_connectors import matrix_mappo_v8_15agv as connector
from thesis.pypeline_connectors.v7_inference import verify_checkpoint

cfg, raw = verify_checkpoint(connector.CHECKPOINT, connector.RUN_CONFIG)
assert int(cfg['fleetsize']) == 15
assert int(cfg['max_fleetsize']) == 20
assert set(raw['roles']) == {'routing', 'dispatch_front', 'dispatch_rear'}
assert connector.get_config('fleetsize') == 15
print('V8 DESKTOP CHECK: PASS')
print('Checkpoint:',connector.CHECKPOINT)
print('Run config:',connector.RUN_CONFIG)
print('Fleetsize:',connector.get_config('fleetsize'))
print('Routing:', cfg['routing_encoder'], 'Dijkstra:',cfg['final_used_guide_strength'])
