"""V17-B 7-edge-conditioned-message GATv2 + unchanged Full-Rollout GA512 MAPPO.

Additive wrapper: existing V15 files stay byte-identical. Separate run root,
new V17-B lineage fingerprints and version. Supports V15 EP0 warm-start for
joint16 via original --init-checkpoint; V17-B native --resume thereafter.
"""
#==================== MODULE 00 - GẮN WRAPPER V17-B VỚI CƠ SỞ V15 ====================
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
from thesis.mappo.v16b_traffic_edge import install_v16b, transfer_v15_checkpoint, EDGE_FEATURE_NAMES

VERSION = 'V17B_EDGE7_MESSAGE_GATV2_FULLROLLOUT_GA512_V1'
DEFAULT_ROOT = ROOT / 'mappo_runs' / 'V17B_EDGE7_MSG_GA512'
EDGE_SOURCE = ROOT / 'thesis' / 'mappo' / 'v16b_traffic_edge.py'
WRAPPER_SOURCE = Path(__file__).resolve()
ORIGINAL_BUILD = builder.build_v15
ORIGINAL_INSTALL = arch15.install_v15



#=============== KẾT THÚC MODULE 00 - GẮN WRAPPER V17-B VỚI CƠ SỞ V15 ===============




#================ MODULE 01 - TẠO MÔ HÌNH V17-B VÀ CHUYỂN CHECKPOINT ================
# Mục đích: Gọi builder V15 nhưng lắp Traffic-GATv2 thay Routing GATv2 cũ.
# Giải thích: Nếu warm-start: chuyển trọng số tương thích, reset Adam Routing Actor khi cần.

def build_traffic(*args, **kwargs):
    """Reuse V15 AnyLogic/env/CTDE/dispatch; install traffic Routing only."""
    kwargs = dict(kwargs)

    #=================== PHẦN BUILD_TRAFFIC-01 - TÁCH CHECKPOINT NGUỒN ===================
    # Tránh dùng V15 MAPPO.load strict trước khi cài kiến trúc V17-B.

    ck = kwargs.pop('checkpoint', None)
    old_install = arch15.install_v15

    #================ PHẦN BUILD_TRAFFIC-02 - TẠM THAY ROUTING ACTOR V15 ================
    # Lắp GAT V17-B mới vào builder V15 hiện hữu.

    arch15.install_v15 = lambda model, obs_space, dim, graph: install_v16b(
        model, obs_space, dim, graph, v15_install=ORIGINAL_INSTALL)
    try:
        # Don't pass V15 checkpoint to strict MAPPO.load before adapter exists.
        env, model, graph, architecture = ORIGINAL_BUILD(*args, checkpoint=None, **kwargs)
    finally:
        arch15.install_v15 = old_install

    #============== PHẦN BUILD_TRAFFIC-03 - CHUYỂN TRỌNG SỐ KHI WARM-START ==============
    # Nạp trọng số V15 Joint16 EP0; reset optimizer Routing Actor.

    if ck is not None:
        try:
            conversion = transfer_v15_checkpoint(model, ck)
            architecture['checkpoint_transfer'] = conversion
            if conversion == 'v15_compatible_weights_routing_actor_adam_reset':
                print('[V17B WARMSTART] V15 actors/critics transferred; '
                      'Routing actor Adam reset (7 edge features and message params); '
                      'Dispatch and critics Adam/ValueNorm preserved.', flush=True)
        except BaseException:
            env.close()
            raise
    return env, model, graph, architecture

#============ KẾT THÚC MODULE 01 - TẠO MÔ HÌNH V17-B VÀ CHUYỂN CHECKPOINT ============






#=================== MODULE 02 - THAY ĐỔI NGỮ CẢNH RUN THÀNH V17-B ===================
# Mục đích: Bổ sung fingerprint của Traffic-GATv2, mô tả edge features, chọn output root V17-B.
# Giải thích: Gọi lại full.main() rồi phục hồi các hàm V15 đã tạm thay trong bộ nhớ.

def main(argv=None):
    old_preflight, old_meta = base.preflight, base._meta
    old_build = builder.build_v15
    old_version, old_full_version, old_default = base.VERSION, full.VERSION, full.DEFAULT_OUT


    #======================== PHẦN MAIN-01 - THÊM FINGERPRINT MỚI ========================
    # Khóa mã nguồn V17-B 7 Edge + Message và wrapper.

    def preflight_v16(args):
        cfg, bp, simulator, fps, out, index = old_preflight(args)
        fps = dict(fps)
        fps['v17b_edge_message_actor'] = base._digest(EDGE_SOURCE)
        fps['v17b_runner'] = base._digest(WRAPPER_SOURCE)
        return cfg, bp, simulator, fps, out, index


    #================== PHẦN MAIN-02 - KHAI BÁO EDGE FEATURES / NO MASK ==================
    # Metadata nêu 7 đặc trưng cạnh / message / Routing không mask.

    def meta_v16(*args):
        m = old_meta(*args)
        m['architecture'] = 'V17B 7-edge conditioned attention+message GATv2 2x4/32 (no mask) + V15 LinFE/CTDE'
        m['edge_features'] = list(EDGE_FEATURE_NAMES)
        m['edge_attention_conditioned'] = True
        m['edge_message_conditioned'] = True
        m['edge_dimension'] = len(EDGE_FEATURE_NAMES)
        m['stationary_occupancy_is_proxy'] = True
        m['traffic_observation_only'] = True
        m['routing_action_mask_applied'] = False
        m['v17b_v15_source_changed'] = False
        return m


    #======================= PHẦN MAIN-03 - TẠM GẮN CẤU HÌNH V17-B =======================
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

#============== KẾT THÚC MODULE 02 - THAY ĐỔI NGỮ CẢNH RUN THÀNH V17-B ==============






#========================= MODULE 03 - ĐIỂM VÀO RUNNER V17-B =========================
# Mục đích: Kích hoạt main() khi người dùng gọi file qua PowerShell.
# Giải thích: Không sửa source V15 trên đĩa.

if __name__ == '__main__':
    main()

#==================== KẾT THÚC MODULE 03 - ĐIỂM VÀO RUNNER V17-B ====================




