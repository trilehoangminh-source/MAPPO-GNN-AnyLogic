#=================== GIẢI THÍCH FILE: THỬ NGHIỆM HEURISTIC DIJKSTRA ===================
# Module định tuyến hướng dẫn Dijkstra của các nhánh cũ; V16-B không bật hướng dẫn này.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================


"""B3 experiment: MAPPO with a *soft*, directed-graph shortest-path prior.

Inspired by Peng (2023), MAPPOHR (arXiv:2306.01270). This module is our
Tobias/AnyLogic adaptation, NOT a reproduction of the paper's full planner.

Critical PPO invariant: exactly the same additive logit prior is stored in
rollout observations and used in PPO's recomputed action distributions.
No intervention is made AFTER selecting an action. The environment action mask
remains authoritative. No privileged critic-only data is fed to the actor.
"""



#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from __future__ import annotations
from collections import Counter, defaultdict
import heapq
from math import hypot, isfinite
from types import MethodType
import numpy as np
import torch
from torch.distributions import Categorical
from .algorithm import MAPPO
from .collector_stop_ab import EventCollector
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================




#================================ MODULE 02 - HÀM _KEY ================================
# Tạo khóa từ trạng thái/toạ độ để tra cứu heuristic.
def _key(coord):
    return tuple(round(float(c), 6) for c in coord)
#=========================== KẾT THÚC MODULE 02 - HÀM _KEY ============================




#========================= MODULE 03 - LỚP DIRECTEDGRAPHGUIDE =========================
# Bộ định hướng theo graph/đích dùng heuristic Dijkstra, không bật ở V16-B.
class DirectedGraphGuide:
    """Suggest closest-path legal next move, using Tobias's directed Hive graph.

    Each 1..4 move maps to the agent-visible neighbor coordinates at agvs[0,8:16].
    Target is agvs[0,6:8]. Edge directions are honored; no directed-edge-free
    Euclidean distance shortcut is used. Dynamic AGVs/blocks remain to MAPPO.
    """


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, hive, strength=1.5):
        self.hive = hive
        self.strength = float(strength)
        if not 0 <= self.strength <= 20:
            raise ValueError('routing guide strength must be between 0 and 20')
        self._cache_token = None
        self._coords_to_id = {}
        self._reverse = {}
        self._dist_cache = {}
        self.stats = Counter()
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #========================== MODULE CON 02 - RESET_STATS ===========================
    # Mục đích: Xử lý reset stats theo logic của lớp DirectedGraphGuide.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def reset_stats(self):
        self.stats.clear()
    #====================== KẾT THÚC MODULE CON 02 - RESET_STATS ======================




    #============================ MODULE CON 03 - _REFRESH ============================
    # Mục đích: Hàm nội bộ refresh; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _refresh(self):
        nodes, paths = self.hive.nodes, self.hive.paths
        if not nodes or not paths:
            return False
        token = (id(nodes), id(paths))
        if token == self._cache_token:
            return True
        self._cache_token = token
        self._coords_to_id = {_key(coords): int(i) for i, coords in nodes.items()}
        self._reverse = defaultdict(list)
        self._dist_cache.clear()
        for u, v in paths:  # directed path list; bidirectional arcs already expanded by Hive
            if int(u) not in nodes or int(v) not in nodes:
                raise ValueError(f'Bad graph arc: {(u,v)}')
            c1, c2 = nodes[u], nodes[v]
            w = hypot(float(c1[0])-float(c2[0]), float(c1[1])-float(c2[1]))
            if w > 0:
                self._reverse[int(v)].append((int(u), w))
        return True
    #======================= KẾT THÚC MODULE CON 03 - _REFRESH ========================




    #======================== MODULE CON 04 - _DIST_TO_TARGET =========================
    # Mục đích: Hàm nội bộ dist to target; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _dist_to_target(self, target_id):
        if target_id not in self._dist_cache:
            distance = {target_id: 0.0}
            todo = [(0.0, target_id)]
            while todo:
                d, node = heapq.heappop(todo)
                if d > distance[node] + 1e-12:
                    continue
                for previous, weight in self._reverse.get(node, ()):
                    cand = d + weight
                    if cand + 1e-12 < distance.get(previous, float('inf')):
                        distance[previous] = cand
                        heapq.heappush(todo, (cand, previous))
            self._dist_cache[target_id] = distance
        return self._dist_cache[target_id]
    #==================== KẾT THÚC MODULE CON 04 - _DIST_TO_TARGET ====================




    #============================ MODULE CON 05 - SUGGEST =============================
    # Mục đích: Tính prior định tuyến dựa trên khoảng cách graph; chỉ dành cho nhánh guided cũ.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def suggest(self, observation):
        """Return 5-element logit prior and diagnostic tag. Never mask moves."""
        self.stats['routing_decisions'] += 1
        prior = np.zeros(5, dtype=np.float32)
        if not self._refresh():
            self.stats['missing_graph'] += 1
            return prior, 'missing_graph'
        agvs = np.asarray(observation['agvs'], dtype=np.float32)
        mask = np.asarray(observation['action_mask'], dtype=np.float32).reshape(-1)[:5]
        if agvs.ndim != 2 or agvs.shape[1] < 16:
            raise ValueError('Routing observation lacks current/target/neighbor coordinates')
        ego = agvs[0]
        current_id = self._coords_to_id.get(_key(ego[4:6]))
        target_id = self._coords_to_id.get(_key(ego[6:8]))
        if current_id is None or target_id is None:
            self.stats['unmapped_coordinates'] += 1
            return prior, 'unmapped_coordinates'
        if current_id == target_id:
            self.stats['at_target'] += 1
            return prior, 'at_target'
        # The four movement action slots are in Tobias's original observation order.
        distances = self._dist_to_target(target_id)
        valid_candidates = []
        for idx in range(1, 5):
            if mask[idx] <= 0.5:
                continue
            next_id = self._coords_to_id.get(_key(ego[8 + (idx-1)*2 : 10 + (idx-1)*2]))
            if next_id is None or next_id not in distances:
                continue
            # Only guide available outgoing arcs, never hallucinate a transition.
            if not any(prev == current_id for prev, _ in self._reverse.get(next_id, ())):
                continue
            cx,cy = self.hive.nodes[current_id]
            nx,ny = self.hive.nodes[next_id]
            local = hypot(float(nx)-float(cx), float(ny)-float(cy))
            valid_candidates.append((idx, distances[next_id] + local))
        if not valid_candidates:
            self.stats['no_legal_path'] += 1
            return prior, 'no_legal_path'
        minimum = min(distance for _,distance in valid_candidates)
        best = [idx for idx,d in valid_candidates if abs(d-minimum)<1e-6]
        for idx in best:
            prior[idx] = self.strength
        self.stats['guided'] += 1
        self.stats['choices_guided'] += len(best)
        return prior, 'guided'
    #======================== KẾT THÚC MODULE CON 05 - SUGGEST ========================

#==================== KẾT THÚC MODULE 03 - LỚP DIRECTEDGRAPHGUIDE =====================




#================= MODULE 04 - HÀM ENABLE_ROUTING_DISTRIBUTION_PRIOR ==================
# Gắn prior Dijkstra cho nhánh thử nghiệm guided cũ.
def enable_routing_distribution_prior(model):
    """Patch ONLY this model's Routing distribution and MAPPO's tensor adapter.

    No changes to neural parameters/architecture, so checkpoints retain the
    normal MAPPO shape. The static guide prior must be supplied via each stored
    routing observation during BOTH sampling and updates/evaluation.
    """
    import thesis.mappo.algorithm as algo
    if not getattr(algo.as_actor, '_b3_guidance', False):
        original_adapter = algo.as_actor
        def _guided_adapter(obs, device):
            result = original_adapter(obs, device)
            if 'routing_guide_bias' in obs:
                result['routing_guide_bias'] = torch.as_tensor(
                    obs['routing_guide_bias'], dtype=torch.float32, device=device)
            return result
        _guided_adapter._b3_guidance = True
        algo.as_actor = _guided_adapter

    actor = model.roles['routing'].actor
    if getattr(actor, '_b3_guide_dist', False):
        return
    original_dist = actor.dist
    def _guided_dist(self, obs):
        if 'routing_guide_bias' not in obs:
            raise RuntimeError('B3 routing observation missing guidance; PPO would be inconsistent')
        dist = original_dist(obs)
        bias = obs['routing_guide_bias']
        if bias.shape != dist.logits.shape:
            raise ValueError(f'Guidance shape {bias.shape} != routing logits {dist.logits.shape}')
        if not torch.isfinite(bias).all():
            raise ValueError('Nonfinite shortest-path bias')
        # Original hard Action Mask still takes priority. Never unmask invalid.
        bias = bias.masked_fill(obs['action_mask'] <= 0.5, 0.0)
        return Categorical(logits=dist.logits + bias)
    actor.dist = MethodType(_guided_dist, actor)
    actor._b3_guide_dist = True
#============= KẾT THÚC MODULE 04 - HÀM ENABLE_ROUTING_DISTRIBUTION_PRIOR =============




#============================ MODULE 05 - LỚP GUIDEDMAPPO =============================
# Biến thể MAPPO có prior định tuyến dùng để so sánh thực nghiệm.
class GuidedMAPPO(MAPPO):


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        enable_routing_distribution_prior(self)
        self.guidance_stats = Counter()
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================== MODULE CON 02 - ACT ===============================
    # Mục đích: Xử lý act theo logic của lớp GuidedMAPPO.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def act(self, role, obs, state, greedy=False):
        action, logp, value = super().act(role, obs, state, greedy)
        if role == 'routing':
            self.guidance_stats['routing_actions'] += 1
            prior = obs['routing_guide_bias']
            if np.any(prior > 0):
                self.guidance_stats['guided_actions'] += 1
                self.guidance_stats['selected_guide'] += int(prior[int(action)] > 0)
        return action, logp, value
    #========================== KẾT THÚC MODULE CON 02 - ACT ==========================

#======================== KẾT THÚC MODULE 05 - LỚP GUIDEDMAPPO ========================




#======================== MODULE 06 - LỚP GUIDEDEVENTCOLLECTOR ========================
# Collector cho biến thể policy có hướng dẫn heuristic.
class GuidedEventCollector(EventCollector):


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, model, environment, stop_mask_mode='forbid_when_move',
                 routing_guide_strength=1.5):
        super().__init__(model, environment, stop_mask_mode)
        self.guide = DirectedGraphGuide(environment.agent_hive, routing_guide_strength)
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================ MODULE CON 02 - _PREPARE ============================
    # Mục đích: Chuẩn bị observation và kiểm tra kích thước/mask.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _prepare(self, observation, role):
        obs = super()._prepare(observation, role)
        if role == 'routing':
            bias, _ = self.guide.suggest(obs)
            obs['routing_guide_bias'] = bias.copy()
        return obs
    #======================= KẾT THÚC MODULE CON 02 - _PREPARE ========================




    #============================ MODULE CON 03 - EPISODE =============================
    # Mục đích: Thu thập một episode và trả trajectory cùng thống kê.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def episode(self, *args, **kwargs):
        self.guide.reset_stats()
        self.model.guidance_stats.clear()
        segments, meta = super().episode(*args, **kwargs)
        stats = dict(self.guide.stats)
        stats.update(dict(self.model.guidance_stats))
        meta['routing_guidance'] = stats
        if stats.get('routing_actions', 0) and stats.get('guided', 0) == 0:
            raise RuntimeError('B3 got no graph path guidance: check networkcontext and coordinate matching; do NOT train blindly')
        return segments, meta
    #======================== KẾT THÚC MODULE CON 03 - EPISODE ========================

#=================== KẾT THÚC MODULE 06 - LỚP GUIDEDEVENTCOLLECTOR ====================


