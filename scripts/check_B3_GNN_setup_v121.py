"""Read-only B3-GNN integration check. No AnyLogic startup or file modification."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
needed=[
 'thesis/mappo/algorithm.py','thesis/mappo/networks.py',
 'thesis/mappo/collector_stop_ab.py','thesis/mappo/routing_guidance_v121.py',
 'thesis/mappo/gnn_routing_B3_v121.py',
 'scripts/train_mappo_long_B3_v121.py',
 'scripts/train_mappo_long_B3_GNN_v121.py',
 'scripts/evaluate_mappo_long_B3_GNN_v121.py',
 'thesis/envs/behaviors_v121.py',
]
missing=[name for name in needed if not (ROOT/name).is_file()]
for name in needed:
    print(('MISSING' if name in missing else 'OK'), name)
if missing:
    raise SystemExit('Missing existing B3/B2 files: patch cannot work on original Tobias-only checkout.')
text=(ROOT/'thesis/envs/behaviors_v121.py').read_text(encoding='utf8',errors='replace')
if 'MAPPO_DISPATCH_MASK_MODE' not in text or 'operation_first' not in text:
    raise SystemExit('B2 operation_first hook not found in behaviors_v121.py; do not overwrite it.')
from thesis.mappo.gnn_routing_B3_v121 import GraphRoutingActor
from thesis.mappo.routing_guidance_v121 import GuidedMAPPO, GuidedEventCollector
print('PASS: B2 operation_first hook + B3 guided collector + new true GNN imports')
