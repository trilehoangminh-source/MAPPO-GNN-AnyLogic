"""Conservative, reversible patch for V8 20-AGV CUDA OOM.
Run from the repo root: python scripts/fix_v8_gpu_oom.py --check/--apply.
Only exact known source patterns are modified. No checkpoints are touched.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
GNN = ROOT/'thesis'/'mappo'/'gnn_routing_B3_v121.py'
PPO = ROOT/'thesis'/'mappo'/'mappo_stability_v7.py'

GNN_CHANGES = (
    ('torch.where(pos_valid, h[b, pos_i[:, 0]].unsqueeze(1),',
     'torch.where(pos_valid.unsqueeze(-1), h[b, pos_i[:, 0]].unsqueeze(1),'),
    ('torch.where(tar_valid, h[b, tar_i[:, 0]].unsqueeze(1),',
     'torch.where(tar_valid.unsqueeze(-1), h[b, tar_i[:, 0]].unsqueeze(1),'),
)
PPO_OLD = '        n_parts = max(1, min(cfg.mini_batches, n))'
PPO_NEW = '''        n_parts = max(1, min(cfg.mini_batches, n))
        # Optional memory safety guard for 4GB CUDA GPUs. The default remains
        # byte-for-byte equivalent to the original PPO minibatch partition.
        # Smaller minibatches add optimizer steps; compare resulting policies
        # by evaluation instead of assuming identical PPO update dynamics.
        max_mini = int(os.environ.get('MAPPO_MAX_MINIBATCH', '0'))
        if max_mini < 0:
            raise ValueError('MAPPO_MAX_MINIBATCH must be >= 0')
        if max_mini:
            n_parts = max(n_parts, math.ceil(n / max_mini))
            if role == 'routing':
                print(f'[V8 VRAM GUARD] routing_samples={n} minibatches={n_parts} max_samples_per_minibatch={math.ceil(n/n_parts)}', flush=True)'''


def check_and_plan():
    if not GNN.is_file() or not PPO.is_file():
        raise FileNotFoundError('Run from root of full thesis project. Missing GNN/V7 PPO file')
    src = GNN.read_text(encoding='utf-8')
    code = PPO.read_text(encoding='utf-8')
    pending_gnn = []
    for original, fixed in GNN_CHANGES:
        cnt_old = src.count(original)
        cnt_new = src.count(fixed)
        if cnt_old == 1 and cnt_new == 0:
            pending_gnn.append((original, fixed))
        elif cnt_old == 0 and cnt_new == 1:
            pass  # already fixed
        else:
            raise RuntimeError('Refuse to patch unexpected GNN source pattern: '+original)
    pending_ppo = False
    if code.count(PPO_OLD) == 1 and code.count('MAPPO_MAX_MINIBATCH') == 0:
        pending_ppo = True
    elif code.count(PPO_NEW) == 1:
        pass
    else:
        raise RuntimeError('Refuse to patch unexpected V7 PPO source pattern')
    return src, code, pending_gnn, pending_ppo


def atomic_write(path, data):
    temp = path.with_name(path.name + '.oom-tmp')
    temp.write_text(data, encoding='utf-8', newline='')
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    a = p.parse_args()
    src, code, changes, patch_ppo = check_and_plan()
    print('GNN broadcasting changes needed:', len(changes))
    print('Optional VRAM minibatch guard needed:', patch_ppo)
    if a.check:
        print('PREFLIGHT PASS (no files changed)')
        return
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for path, old_content, do_patch in ((GNN, src, bool(changes)), (PPO, code, patch_ppo)):
        if do_patch:
            backup = path.with_name(path.name + '.pre_gpu_oom_' + stamp + '.bak')
            backup.write_bytes(path.read_bytes())
            print('Backup:', backup)
    for original, fixed in changes:
        src = src.replace(original, fixed, 1)
    if changes:
        atomic_write(GNN, src)
    if patch_ppo:
        # mappo_stability_v7.py already imports math; add os explicitly.
        if '\nimport os\n' not in code:
            code = code.replace('import math\n', 'import math\nimport os\n', 1)
        code = code.replace(PPO_OLD, PPO_NEW, 1)
        atomic_write(PPO, code)
    check_and_plan()
    print('PATCH PASS: GNN broadcasting safe; PPO VRAM guard available.')
    print('Set $env:MAPPO_MAX_MINIBATCH="64" before CUDA training to activate the guard.')

if __name__ == '__main__':
    main()
