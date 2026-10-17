"""Reject wrong installation before expensive AnyLogic B3 training."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
checks={
    'Existing MAPPO algorithm': ROOT/'thesis/mappo/algorithm.py',
    'Existing pooled Actor': ROOT/'thesis/mappo/networks.py',
    'Existing B2 patch hook': ROOT/'thesis/envs/behaviors_v121.py',
    'Existing operation-first decoder': ROOT/'thesis/envs/dispatch_paper_mask_v121.py',
    'Existing fresh Station model': ROOT/'envs/Matrix_critic_fresh.zip',
    'New B3 graph guide': ROOT/'thesis/mappo/routing_guidance_v121.py',
    'New B3 trainer': ROOT/'scripts/train_mappo_long_B3_v121.py',
}
failed=False
for label,path in checks.items():
    good=path.is_file()
    print(f'{"OK" if good else "MISSING"}: {label} -> {path.relative_to(ROOT)}')
    failed |= not good
b2_source=ROOT/'thesis/envs/behaviors_v121.py'
if b2_source.is_file():
    t=b2_source.read_text(encoding='utf-8')
    if 'MAPPO_DISPATCH_MASK_MODE' not in t or 'build_operation_first_dispatch_mask' not in t:
        print('ERROR: B2 operation-first hook not installed. Do not proceed.'); failed=True
if failed: raise SystemExit('B3 setup failed. Restore existing B2 setup first; no files changed.')
print('PASS: B3 files and existing B2 hook found; live graph coordinates still require smoke test.')
