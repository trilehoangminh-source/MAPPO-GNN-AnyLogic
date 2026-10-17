"""Offline smoke and safety tests. python tests/test_dispatch_paper_mask_v121.py"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from thesis.envs.part_variants import MatrixPart
from thesis.envs.dispatch_paper_mask_v121 import build_operation_first_dispatch_mask as build
p=MatrixPart()
live=[4,9,14,19,24,29,34,39,40,41,42,43,44,49,54,67,68,69,74,83,84,95,96,97,98,99,104]

def check(info, expected_nodes, why):
    mask,meta=build(info,p,live)
    nodes={live[i] for i,x in enumerate(mask) if x>0.5}
    assert nodes==set(expected_nodes),(why,nodes,meta)
    assert mask.sum()>0,why
    return meta

m=check(None,[67,104],'inbound pickup')
assert not m['buffer_fallback']
info={'variant':'vb_1','rework':False,'next_geo':None,'proc':['HSN1','WPS'],'amount':[11,0]}
check(info,[29,41],'operation HSN1')
info['next_geo']=2
check(info,[68],'next GEO')
info['rework']=True
check(info,[39],'rework')
info={'variant':'hc_1','rework':False,'next_geo':None,'proc':['HSN1'],'amount':[0]}
m=check(info,[95,96,97,98,99],'fallback buffer')
assert m['buffer_fallback']
# No legitimate stationary action should become an invalid no-op mask.
info={'variant':'hc_1','rework':False,'next_geo':6,'proc':[],'amount':[]}
check(info,[95,96,97,98,99],'missing GEO fallback')
print('PASS: B2 dispatch priority, target-node order, fallback, legal nonempty masks')
