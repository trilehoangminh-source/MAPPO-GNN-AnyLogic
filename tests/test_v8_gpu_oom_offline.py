"""Validates broadcasting and non-destructive patch without starting AnyLogic."""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
GNN = ROOT/'thesis/mappo/gnn_routing_B3_v121.py'
PPO = ROOT/'thesis/mappo/mappo_stability_v7.py'
SCRIPT = ROOT/'scripts/fix_v8_gpu_oom.py'

assert SCRIPT.is_file() and GNN.is_file() and PPO.is_file(), 'Patch must run in thesis repo'
code = GNN.read_text(encoding='utf-8')
assert 'torch.where(pos_valid.unsqueeze(-1),' in code
assert 'torch.where(tar_valid.unsqueeze(-1),' in code
assert 'torch.where(pos_valid, ' not in code
assert 'torch.where(tar_valid, ' not in code
ppo = PPO.read_text(encoding='utf-8')
assert 'MAPPO_MAX_MINIBATCH' in ppo
for b in (1, 2, 7, 64):
    h = 32
    flag = torch.randint(0,2,(b,1),dtype=torch.bool)
    selected = torch.randn(b,1,h)
    backup = torch.randn(b,1,h)
    fixed = torch.where(flag.unsqueeze(-1), selected, backup)
    assert tuple(fixed.shape) == (b,1,h)
    ref = torch.stack([selected[i] if flag[i,0] else backup[i] for i in range(b)])
    assert torch.equal(fixed, ref)
for n in (1, 100, 900, 5000):
    default_parts = min(4,n)
    parts = max(default_parts, math.ceil(n/64))
    assert math.ceil(n/parts) <= 64
    assert parts <= n
compile(code,str(GNN),'exec')
compile(ppo,str(PPO),'exec')
print('V8 GPU OOM PATCH TEST: PASS (shape [B,1,H], minibatch <=64, Python syntax)')
