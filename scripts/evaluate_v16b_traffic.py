"""Unmodified V15 held-out runner applied to V16-B actor/checkpoints.

10 seeds greedy by default; no learning, no checkpoint modification.
Uses a separate V16-B run root and verifies V16-B architecture fingerprints.
"""
#================ MODULE 00 - GẮN HELD-OUT V16-B VỚI BỘ EVALUATOR V15 ================
# Mục đích: Import các hàm từ V15 nhưng cho phép dùng đúng Traffic-GATv2 của V16-B.
# Giải thích: Không định nghĩa thuật toán PPO mới, không thay đổi checkpoint.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import evaluate_v15_tobias_staged as eval15
import v15_tobias_build as builder
import train_v16b_fullrollout as v16
import train_v15_tobias_staged as base



#=========== KẾT THÚC MODULE 00 - GẮN HELD-OUT V16-B VỚI BỘ EVALUATOR V15 ===========




#================ MODULE 01 - CHỐT CHECKPOINT V16-B VÀ GỌI EVALUATOR ================
# Mục đích: Chặn việc đọc nhầm checkpoint V15 bằng version và fingerprint V16-B.
# Giải thích: Tạm gắn hàm build_traffic, gọi evaluator V15, sau đó phục hồi hàm cũ.

def main(argv=None):

    #===================== PHẦN MAIN-01 - ĐỌC THAM SỐ, ĐẶT RUN V16-B =====================
    # Mặc định đọc mappo_runs/V16B_TRAFFIC_EDGE_GATV2_GA512.

    argv = list(sys.argv[1:] if argv is None else argv)
    if '--output-root' not in argv:
        argv += ['--output-root', str(v16.DEFAULT_ROOT)]
    # Evaluate V16-B only (no V15 directory / accidental checkpoint mismatch).
    out = Path(argv[argv.index('--output-root') + 1]).resolve()
    phase = argv[argv.index('--phase') + 1] if '--phase' in argv else 'joint20'
    meta = base._read(out / phase / 'experiment.json')

    #======================= PHẦN MAIN-02 - KIỂM TRA VERSION V16-B =======================
    # Không để evaluator nhận nhầm checkpoint V15.

    if meta.get('version') != v16.VERSION:
        raise RuntimeError('Not a V16-B run; refusing incorrect architecture/checkpoint')
    for key, source in [('v16b_edge_traffic_actor', v16.EDGE_SOURCE),
                        ('v16b_runner', v16.WRAPPER_SOURCE)]:
        if meta['fingerprints'].get(key) != base._digest(source):
            raise RuntimeError(f'V16B source changed: {key}')

    #======================= PHẦN MAIN-03 - THAY BUILDER TẠM THỜI =======================
    # Dùng Traffic Routing Actor rồi trả lại builder V15 sau đánh giá.

    old_build = builder.build_v15
    builder.build_v15 = v16.build_traffic
    try:

        #=============== PHẦN MAIN-04 - GỌI HELD-OUT V15 DÙNG KIẾN TRÚC V16-B ===============
        # Evaluator đã có sẵn: không viết PPO mới.

        return eval15.main(argv)
    finally:
        builder.build_v15 = old_build

#============ KẾT THÚC MODULE 01 - CHỐT CHECKPOINT V16-B VÀ GỌI EVALUATOR ============






#======================= MODULE 02 - ĐIỂM VÀO EVALUATION V16-B =======================
# Mục đích: Gọi main() khi thực thi file trực tiếp.
# Giải thích: Kết quả held-out vẫn được lưu ở run root V16-B riêng.

if __name__ == '__main__':
    main()

#================== KẾT THÚC MODULE 02 - ĐIỂM VÀO EVALUATION V16-B ==================




