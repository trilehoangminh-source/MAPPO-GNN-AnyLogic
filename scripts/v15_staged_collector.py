"""Staged-training behavior adapter for V15, no PPO/collector modifications.

One unmodified EventCollector still connects actions to reward transitions.
Only active-role trajectories are handed to the unmodified PPO trainer.
Frozen roles have explicitly defined non-learning decisions:
  Routing-only:  random LEGAL dispatch station/buffer (Tobias pretraining idea)
  Dispatch-only: native Tobias CollisionFreeRouting in Matrix(pseudo_routing=True).
  This is an environment-owned, NON-learning controller; it NEVER enters PPO.
  Learned Routing GAT is used only in routing8/joint*, without Dijkstra bias.
"""
#=================== MODULE 00 - KHAI BÁO PHASE VÀ ROLE HUẤN LUYỆN ===================
# Mục đích: Định nghĩa số AGV và các role được cập nhật trong từng phase.
# Giải thích: routing8/dispatch8 pretrain riêng; joint* train đồng thời ba role.

from __future__ import annotations
import numpy as np


PHASES={
    'routing8': (8,('routing',)),
    'dispatch8': (8,('dispatch_front','dispatch_rear')),
    'joint8': (8,('routing','dispatch_front','dispatch_rear')),
    'joint12': (12,('routing','dispatch_front','dispatch_rear')),
    'joint16': (16,('routing','dispatch_front','dispatch_rear')),
    'joint20': (20,('routing','dispatch_front','dispatch_rear')),
}



#============== KẾT THÚC MODULE 00 - KHAI BÁO PHASE VÀ ROLE HUẤN LUYỆN ==============




#==================== MODULE 01 - CHỌN DISPATCH NGẪU NHIÊN HỢP LỆ ====================
# Mục đích: Lấy các action Dispatch có mask > 0.5 và random trong tập hợp lệ.
# Giải thích: Chỉ dùng khi Dispatch bị đóng băng; Routing learned vẫn không mask.

def masked_random_dispatch(obs, rng):
    mask=np.asarray(obs['action_mask'],dtype=np.float32)
    valid=np.flatnonzero(mask>.5)
    if valid.size<1: raise RuntimeError('No legal Dispatch action')
    return int(rng.choice(valid))

#=============== KẾT THÚC MODULE 01 - CHỌN DISPATCH NGẪU NHIÊN HỢP LỆ ===============






#================== MODULE 02 - BỘ CHUYỂN ĐỔI QUYẾT ĐỊNH THEO PHASE ==================
# Mục đích: Nếu role active: dùng Actor MAPPO. Nếu Dispatch chưa train: random hợp lệ.
# Giải thích: Nếu dispatch8: Routing dùng CollisionFreeRouting của môi trường, không PPO.

class StagePolicyAdapter:
    """Use an original MAPPO model for the trained roles, heuristics otherwise."""

    #=================== PHẦN STAGEPOLICYADAPTER-01 - KHỞI TẠO ADAPTER ===================
    # Lưu mô hình, role train và RNG cho dispatch giả lập.

    def __init__(self,model,phase,seed):
        if phase not in PHASES: raise ValueError('Unknown V15 phase')
        self.model=model;self.roles=model.roles;self.cfg=model.cfg
        self.active=set(PHASES[phase][1]);self.phase=phase
        self.rng=np.random.default_rng(seed)


    #================ PHẦN STAGEPOLICYADAPTER-02 - ÁNH XẠ AGENT SANG ROLE ================
    # Nhờ MAPPO xác định Routing/Dispatch theo ID và đội xe.

    def role_for(self,*args,**kwargs): return self.model.role_for(*args,**kwargs)


    #============== PHẦN STAGEPOLICYADAPTER-03 - CHỌN HÀNH ĐỘNG THEO PHASE ==============
    # Active dùng policy; Dispatch frozen chọn ngẫu nhiên hợp lệ.

    def act(self,role,obs,state,greedy=False):
        if role in self.active:
            return self.model.act(role,obs,state,greedy=greedy)
        if role=='routing':
            raise RuntimeError(
                "dispatch8 must use native Tobias CollisionFreeRouting via "
                "Matrix(pseudo_routing=True). Routing must not reach the MAPPO collector."
            )
        else:
            action=masked_random_dispatch(obs,self.rng)
        # Not trained: collector logs but never passes these records to PPO.
        return action,0.0,0.0


    #=============== PHẦN STAGEPOLICYADAPTER-04 - ƯỚC LƯỢNG VALUE CHO ROLE ===============
    # Role frozen trả 0, active dùng Critic.

    def value(self,role,state):
        return self.model.value(role,state) if role in self.active else 0.0

#============= KẾT THÚC MODULE 02 - BỘ CHUYỂN ĐỔI QUYẾT ĐỊNH THEO PHASE =============






#================== MODULE 03 - LỌC TRAJECTORY THEO ROLE ĐƯỢC TRAIN ==================
# Mục đích: Chỉ chọn segment của role active, chống trộn role trong một segment.
# Giải thích: Dữ liệu role khác không được dùng để PPO update.

def trained_trajectories(trajectories,phase):
    if phase not in PHASES: raise ValueError('Unknown V15 phase')
    allowed=set(PHASES[phase][1])
    out=[]
    for seg in trajectories:
        if not seg: continue
        if any(rec['role']!=seg[0]['role'] for rec in seg):
            raise RuntimeError('Mixed-role lifetime segment: cannot safely train')
        if seg[0]['role'] in allowed: out.append(seg)
    return out

#============= KẾT THÚC MODULE 03 - LỌC TRAJECTORY THEO ROLE ĐƯỢC TRAIN =============




