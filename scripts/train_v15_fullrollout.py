"""Separate V15 full-rollout gradient-accumulation runner.

Uses the original staged trainer for logging, checkpointing and AnyLogic.
Changes ONLY each model's PPO train method. Baseline training sources stay
byte-identical, so V15 1024/4096 baselines can resume using their OLD runner.

Usage: python -u scripts/train_v15_fullrollout.py --phase routing8 ...
       --microbatch-size 512 --max-episodes 1
"""
#============== MODULE 00 - KHAI BÁO FULL-ROLLOUT GRADIENT ACCUMULATION ==============
# Mục đích: Tái sử dụng V15 trainer, gắn bộ optimizer full-rollout xử lý microbatch.
# Giải thích: Batch lớn được tích lũy gradient; microbatch 512 không phải episode.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import train_v15_tobias_staged as base
import v15_tobias_build as builder
from thesis.mappo.v15_fullrollout_accum import enable_fullrollout

OPTIMIZER_SOURCE = ROOT / 'thesis' / 'mappo' / 'v15_fullrollout_accum.py'
DEFAULT_OUT = ROOT / 'mappo_runs' / 'V15_FULLROLLOUT_GA512'
VERSION = 'V15_TOBIAS_FULL_ROLLOUT_GRADACC_V1'



#========= KẾT THÚC MODULE 00 - KHAI BÁO FULL-ROLLOUT GRADIENT ACCUMULATION =========




#=================== MODULE 01 - BỘ CHẠY TRAINING FULL-ROLLOUT V15 ===================
# Mục đích: Bọc (wrapper) trainer V15 bằng các hàm phụ để dùng đúng optimizer GA512.
# Giải thích: Tạm thay hàm trong bộ nhớ Python, sau cùng phục hồi hàm gốc.

def main(argv=None):

    #======================= PHẦN MAIN-01 - NHẬN MICRO-BATCH SIZE =======================
    # Thường 512; dùng để tích lũy gradient full-rollout.

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--microbatch-size', type=int, default=512)
    a, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if a.microbatch_size < 1:
        parser.error('--microbatch-size must be >= 1')
    if '--max-minibatch' in rest:
        parser.error('Use --microbatch-size (not --max-minibatch) in the full-rollout runner')
    if '--output-root' not in rest:
        rest.extend(['--output-root', str(DEFAULT_OUT)])
    # The original runner's `max_minibatch` metadata field is reused as an
    # exact size/lineage guard; its VRAM guard is NOT used by this optimizer.
    rest.extend(['--max-minibatch', str(a.microbatch_size)])

    old_preflight = base.preflight
    old_meta = base._meta
    old_build = builder.build_v15
    base.VERSION = VERSION


    #===================== PHẦN MAIN-02 - GẮN FINGERPRINT OPTIMIZER =====================
    # Xác thực microbatch, code và điều kiện thí nghiệm.

    def preflight_with_accum(args):
        cfg, blueprint, model, fingerprints, out, index = old_preflight(args)
        if int(cfg['mini_batches']) != 1:
            raise ValueError('Full-rollout requires blueprint mini_batches=1')
        fingerprints = dict(fingerprints)
        fingerprints['full_rollout_optimizer'] = base._digest(OPTIMIZER_SOURCE)
        fingerprints['full_rollout_runner'] = base._digest(Path(__file__).resolve())
        return cfg, blueprint, model, fingerprints, out, index


    #======================== PHẦN MAIN-03 - THÊM METADATA GA512 ========================
    # Ghi chế độ optimizer full-rollout và microbatch.

    def meta_with_accum(*args):
        meta = old_meta(*args)
        meta['ppo_update_mode'] = 'full_rollout_gradient_accumulation'
        meta['ppo_microbatch_size'] = a.microbatch_size
        return meta


    #===================== PHẦN MAIN-04 - BẬT OPTIMIZER FULL-ROLLOUT =====================
    # enable_fullrollout thay hành vi update của MAPPO trong bộ nhớ.

    def build_with_accum(*args, **kwargs):
        env, model, graph, arch = old_build(*args, **kwargs)
        try:
            extra = enable_fullrollout(model, microbatch_size=a.microbatch_size)
        except Exception:
            env.close()
            raise
        return env, model, graph, {**arch, **extra}

    # All patches are local to THIS Python process; baseline files on disk
    # are not changed and the baseline executable keeps its original semantics.

    #======================== PHẦN MAIN-05 - TẠM GẮN WRAPPER V15 ========================
    # Chỉ ảnh hưởng tiến trình Python này.

    base.preflight = preflight_with_accum
    base._meta = meta_with_accum
    builder.build_v15 = build_with_accum
    try:

        #====================== PHẦN MAIN-06 - BẮT ĐẦU VÒNG LẶP EPISODE ======================
        # Gọi trainer V15 để collect, update PPO, lưu checkpoint.

        return base.run(rest)
    finally:
        base.preflight = old_preflight
        base._meta = old_meta
        builder.build_v15 = old_build

#============== KẾT THÚC MODULE 01 - BỘ CHẠY TRAINING FULL-ROLLOUT V15 ==============






#========================= MODULE 02 - ĐIỂM VÀO FULL-ROLLOUT =========================
# Mục đích: Chạy main() khi file được gọi từ terminal.
# Giải thích: Không sửa source trên đĩa.

if __name__ == '__main__':
    main()

#==================== KẾT THÚC MODULE 02 - ĐIỂM VÀO FULL-ROLLOUT ====================




