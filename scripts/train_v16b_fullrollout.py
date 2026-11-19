"""V16-B Traffic-Aware Edge-GATv2 + unmodified Full-Rollout GA512 MAPPO.

Additive wrapper: existing V15 files stay byte-identical. Separate run root,
new lineage fingerprints and version. Supports V15 snapshot warm-start for
joint16 via original --init-checkpoint; V16-B native --resume thereafter.
"""
#==================== MODULE 00 - GẮN WRAPPER V16-B VỚI CƠ SỞ V15 ====================
# Mục đích: Nạp V15 Full-Rollout runner và mô hình Traffic-GATv2.
# Giải thích: Không thay đổi bộ tối ưu PPO hay phần AnyLogic có sẵn.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import train_v15_tobias_staged as base
import train_v15_fullrollout as full
import v15_tobias_build as builder
import thesis.mappo.v15_tobias_arch as arch15
from thesis.mappo.v16b_traffic_edge import install_v16b, transfer_v15_checkpoint

VERSION = 'V16B_TRAFFIC_EDGE_GATV2_FULLROLLOUT_GA512_V1'
DEFAULT_ROOT = ROOT / 'mappo_runs' / 'V16B_TRAFFIC_EDGE_GATV2_GA512'
EDGE_SOURCE = ROOT / 'thesis' / 'mappo' / 'v16b_traffic_edge.py'
WRAPPER_SOURCE = Path(__file__).resolve()
ORIGINAL_BUILD = builder.build_v15
ORIGINAL_INSTALL = arch15.install_v15



#=============== KẾT THÚC MODULE 00 - GẮN WRAPPER V16-B VỚI CƠ SỞ V15 ===============




#================ MODULE 01 - TẠO MÔ HÌNH V16-B VÀ CHUYỂN CHECKPOINT ================
# Mục đích: Gọi builder V15 nhưng lắp Traffic-GATv2 thay Routing GATv2 cũ.
# Giải thích: Nếu warm-start: chuyển trọng số tương thích, reset Adam Routing Actor khi cần.

def build_traffic(*args, **kwargs):
    """Reuse V15 AnyLogic/env/CTDE/dispatch; install traffic Routing only."""
    kwargs = dict(kwargs)

    #=================== PHẦN BUILD_TRAFFIC-01 - TÁCH CHECKPOINT NGUỒN ===================
    # Tránh dùng V15 MAPPO.load strict trước khi cài kiến trúc V16-B.

    ck = kwargs.pop('checkpoint', None)
    old_install = arch15.install_v15

    #================ PHẦN BUILD_TRAFFIC-02 - TẠM THAY ROUTING ACTOR V15 ================
    # Gắn hàm install_v16b trong lúc gọi builder V15.

    arch15.install_v15 = lambda model, obs_space, dim, graph: install_v16b(
        model, obs_space, dim, graph, v15_install=ORIGINAL_INSTALL)
    try:
        # Don't pass V15 checkpoint to strict MAPPO.load before adapter exists.
        env, model, graph, architecture = ORIGINAL_BUILD(*args, checkpoint=None, **kwargs)
    finally:
        arch15.install_v15 = old_install

    #============== PHẦN BUILD_TRAFFIC-03 - CHUYỂN TRỌNG SỐ KHI WARM-START ==============
    # Nạp trọng số cũ; reset optimizer Routing Actor nếu thêm edge params.

    if ck is not None:
        try:
            conversion = transfer_v15_checkpoint(model, ck)
            architecture['checkpoint_transfer'] = conversion
            if conversion == 'v15_compatible_weights_routing_actor_adam_reset':
                print('[V16B WARMSTART] V15 actors/critics transferred; '
                      'Routing actor Adam reset (new edge params); '
                      'Dispatch and critics Adam/ValueNorm preserved.', flush=True)
        except BaseException:
            env.close()
            raise
    return env, model, graph, architecture

#============ KẾT THÚC MODULE 01 - TẠO MÔ HÌNH V16-B VÀ CHUYỂN CHECKPOINT ============






#=================== MODULE 02 - THAY ĐỔI NGỮ CẢNH RUN THÀNH V16-B ===================
# Mục đích: Bổ sung fingerprint của Traffic-GATv2, mô tả edge features, chọn output root V16-B.
# Giải thích: Gọi lại full.main() rồi phục hồi các hàm V15 đã tạm thay trong bộ nhớ.

def main(argv=None):
    old_preflight, old_meta = base.preflight, base._meta
    old_build = builder.build_v15
    old_version, old_full_version, old_default = base.VERSION, full.VERSION, full.DEFAULT_OUT


    #======================== PHẦN MAIN-01 - THÊM FINGERPRINT MỚI ========================
    # Khóa mã nguồn Traffic Edge-GATv2 và wrapper V16-B.

    def preflight_v16(args):
        cfg, bp, simulator, fps, out, index = old_preflight(args)
        fps = dict(fps)
        fps['v16b_edge_traffic_actor'] = base._digest(EDGE_SOURCE)
        fps['v16b_runner'] = base._digest(WRAPPER_SOURCE)
        return cfg, bp, simulator, fps, out, index


    #================== PHẦN MAIN-02 - KHAI BÁO EDGE FEATURES / NO MASK ==================
    # Metadata nêu 4 đặc trưng cạnh và hành vi Routing không mask.

    def meta_v16(*args):
        m = old_meta(*args)
        m['architecture'] = 'V16B traffic edge GATv2 2x4/32 (no mask) + V15 LinFE/CTDE'
        m['edge_features'] = [
            'normalized_geometric_length',
            'other_same_direction_flow_per_20',
            'other_opposite_direction_flow_per_20',
            'other_downstream_next_node_pressure_per_20',
        ]
        m['traffic_observation_only'] = True
        m['routing_action_mask_applied'] = False
        m['v16b_v15_source_changed'] = False
        return m


    #======================= PHẦN MAIN-03 - TẠM GẮN CẤU HÌNH V16-B =======================
    # Thay hàm tại runtime trong process Python, không ghi đè V15.

    base.preflight = preflight_v16
    base._meta = meta_v16
    builder.build_v15 = build_traffic
    full.VERSION = VERSION
    full.DEFAULT_OUT = DEFAULT_ROOT
    try:
        # full.main installs unchanged PPO Full-Rollout GA512 implementation.

        #====================== PHẦN MAIN-04 - GỌI TRAINER FULL-ROLLOUT ======================
        # PPO GA512 và vòng lặp episode vẫn theo core V15.

        return full.main(argv)
    finally:
        base.preflight = old_preflight
        base._meta = old_meta
        base.VERSION = old_version
        builder.build_v15 = old_build
        full.VERSION = old_full_version
        full.DEFAULT_OUT = old_default

#============== KẾT THÚC MODULE 02 - THAY ĐỔI NGỮ CẢNH RUN THÀNH V16-B ==============






#========================= MODULE 03 - ĐIỂM VÀO RUNNER V16-B =========================
# Mục đích: Kích hoạt main() khi người dùng gọi file qua PowerShell.
# Giải thích: Không sửa source V15 trên đĩa.

if __name__ == '__main__':
    main()

#==================== KẾT THÚC MODULE 03 - ĐIỂM VÀO RUNNER V16-B ====================




