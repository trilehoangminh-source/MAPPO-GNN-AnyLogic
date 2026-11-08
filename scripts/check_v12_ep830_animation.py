"""Read-only preliminary check for V12 EP830 AnyLogic desktop animation."""
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[1]
run = root / 'mappo_runs' / 'V12_FULL_NO_DIJKSTRA_FROM_V11'
manifest = run / 'experiment.json'
checkpoint = run / 'snapshot_ep_0830.pt'
connector = root / 'thesis' / 'pypeline_connectors' / 'matrix_mappo_v12_20agv.py'
print('ROOT:', root)
for path in (manifest, checkpoint, connector):
    print('OK' if path.is_file() else 'MISSING', path)
    if not path.is_file(): sys.exit(2)
metadata = json.loads(manifest.read_text(encoding='utf-8-sig'))
if metadata.get('version')!='V12_FULL_NO_DIJKSTRA_FROM_V11':
    raise RuntimeError('Wrong experiment metadata')
for key in ('v8_source_checkpoint', 'v8_config_file'):
    p=Path(metadata[key]).resolve()
    print(key,':',p,'FOUND',p.is_file())
    if not p.is_file(): sys.exit(2)
import torch
raw = torch.load(checkpoint,map_location='cpu',weights_only=True)
if set(raw['roles'])!={'routing','dispatch_front','dispatch_rear'}:
    raise RuntimeError('Expected 3 MAPPO roles')
routing = raw['roles']['routing']['actor']
if any('next_hop' in name for name in routing):
    raise RuntimeError('Checkpoint still uses shortest path planning')
if not all(f'layers.{layer}.attention.traffic_score.weight' in routing for layer in (0,1)):
    raise RuntimeError('V12 learned traffic branches missing')
print('OK: V12 3 role actor weights and traffic branches; no next_hop')
print('Ready to set Main.RL_pypeline_name to matrix_mappo_v12_20agv and Simulation runmode=3.')
