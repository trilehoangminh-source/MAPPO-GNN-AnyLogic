#============ GIẢI THÍCH FILE: TRAFFIC-AWARE EDGE-GATV2 - TRỌNG TÂM V16-B =============
# Tính edge_attr từ observation và thêm thành phần cạnh vào attention score; giữ Routing không mask.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================


"""V16-B Traffic-Aware Edge-GATv2, additive to the exact V15 architecture.

Observed AGV fields (Tobias MultiAgent._state_dict):
  agvs[...,0] in_system; [...,1] moving; [...,2:4] last_node;
  [...,4:6] next_node; [...,6:8] target_node.

Four causal, normalized per-directed-edge features (from actor-visible obs only):
  0 geometrical length / max network-arc length
  1 number of OTHER moving AGVs on sender->receiver / 20
  2 number of OTHER moving AGVs on receiver->sender / 20
  3 number of OTHER in-system AGVs with next_node == receiver / 20

All features use only current actor observations; no global privileged state,
future simulator information, or conventional-routing reservations.  Message
passing retains Tobias's original 2-layer/4-head/32-dimensional architecture.
Action distribution remains UNMASKED as in V15/Tobias.
"""



#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from thesis.mappo.v15_tobias_arch import TobiasGATv2, TobiasGraphRoutingActor
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================




#======================= MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ========================
# Các biến module được đọc bởi các hàm/lớp bên dưới; gần với biến toàn cục trong C.
EDGE_FEATURE_NAMES = (
    'normalized_geometric_length',
    'other_same_direction_flow_per_20',
    'other_opposite_direction_flow_per_20',
    'other_downstream_next_node_pressure_per_20',
)
EDGE_DIM = len(EDGE_FEATURE_NAMES)
NORMALIZATION_AGV_SLOTS = 20
#=================== KẾT THÚC MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ===================




#============================ MODULE 03 - LỚP TRAFFICGATV2 ============================
# GATv2 V16-B: thêm W_edge · edge_attr vào attention score; không thay message value.
class TrafficGATv2(TobiasGATv2):
    """Tobias GATv2 + edge term in ATTENTION SCORE (not in message value).

    Zero initialization ensures an upgraded actor initially computes EXACTLY
    the V15 GAT scores, despite its expanded parameterization.
    """



    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo phép chiếu edge_attr thành vector theo từng head; trọng số edge = 0 để lúc đầu giống V15.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, embedding: int = 32, heads: int = 4):
        super().__init__(embedding=embedding, heads=heads)
        self.edge_lin = nn.Linear(EDGE_DIM, embedding * heads, bias=False)
        nn.init.zeros_(self.edge_lin.weight)
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================ MODULE CON 02 - FORWARD =============================
    # Mục đích: Tính attention: W_dst·h_j + W_src·h_i + W_edge·e_ij; softmax theo node nhận rồi gom thông điệp.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        rec, send = edge_index[0], edge_index[1]
        count = rec.numel()
        if edge_attr.shape != (b, count, EDGE_DIM):
            raise ValueError(f'Invalid edge_attr shape {tuple(edge_attr.shape)}, '
                             f'expected {(b, count, EDGE_DIM)}')

        #======================== MỤC NHỎ - CHIẾU NODE VÀ CẠNH ========================
        # Chiếu đặc trưng thành tensor theo attention head.
        left = self.linear_dst(x).view(b, n, h, d)
        right = self.linear_src(x).view(b, n, h, d)
        edge_emb = self.edge_lin(edge_attr).view(b, count, h, d)

        #========================= MỤC NHỎ - ATTENTION SCORE ==========================
        # Thêm embedding cạnh vào score trước LeakyReLU và attention vector.
        pair = nn.functional.leaky_relu(left[:, rec] + right[:, send] + edge_emb, .2)
        scores = (pair * self.att[None, None]).sum(-1).permute(0, 2, 1)
        index = rec.view(1, 1, count).expand(b, h, -1)
        with torch.no_grad():
            maximum = torch.full((b, h, n), -float('inf'), dtype=scores.dtype, device=x.device)
            maximum.scatter_reduce_(2, index, scores.detach(), reduce='amax', include_self=True)

        #====================== MỤC NHỎ - SOFTMAX THEO RECEIVER =======================
        # Chuẩn hóa trọng số attention của mọi cạnh cùng trỏ vào một node.
        weights = torch.exp(scores - maximum.gather(2, index))
        sums = torch.zeros((b, h, n), dtype=weights.dtype, device=x.device)
        sums.scatter_add_(2, index, weights)
        weights = weights / sums.gather(2, index).clamp_min(1e-12)
        sources = right[:, send].permute(0, 2, 1, 3)

        #========================= MỤC NHỎ - MESSAGE PASSING ==========================
        # Nhân thông điệp node gửi với attention rồi scatter_add vào node nhận.
        weighted = sources * weights.unsqueeze(-1)
        result = torch.zeros((b, h, n, d), dtype=x.dtype, device=x.device)
        result.scatter_add_(2, index.unsqueeze(-1).expand(-1, -1, -1, d), weighted)
        return result.mean(1) + self.bias
    #======================== KẾT THÚC MODULE CON 02 - FORWARD ========================

#======================= KẾT THÚC MODULE 03 - LỚP TRAFFICGATV2 ========================




#======================== MODULE 04 - LỚP TRAFFICROUTINGACTOR =========================
# Routing V16-B: kế thừa encoder V15, tính 4 đặc trưng cạnh rồi tạo logits.
class TrafficRoutingActor(TobiasGraphRoutingActor):
    """Drop-in V15 Routing actor with causal traffic-aware edge attention."""
    action_dim = 5
    uses_action_mask = False



    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Thay hai lớp GAT V15 bằng TrafficGATv2, nạp trọng số tương thích và ghi chiều dài cạnh cố định.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, graph, embedding: int = 32, heads: int = 4, layers: int = 2):
        super().__init__(graph, embedding=embedding, heads=heads, layers=layers)
        old = self.convs
        self.convs = nn.ModuleList([TrafficGATv2(embedding, heads) for _ in range(layers)])
        for new, prev in zip(self.convs, old):
            missing, extra = new.load_state_dict(prev.state_dict(), strict=False)
            if set(missing) != {'edge_lin.weight'} or extra:
                raise RuntimeError(f'Bad V15 GAT transfer: missing={missing}, extra={extra}')
        src, dst = self.edge_index[1], self.edge_index[0]
        lengths = torch.linalg.vector_norm(self.node_xy[src] - self.node_xy[dst], dim=-1)
        nonself = src != dst
        real = lengths[nonself]
        max_len = real.max().clamp_min(1e-6) if real.numel() else lengths.new_tensor(1.)
        self.register_buffer('edge_lengths_norm', (lengths / max_len) * nonself.to(lengths.dtype), persistent=False)
        self.register_buffer('edge_nonself', nonself, persistent=False)
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #======================== MODULE CON 02 - TRAFFIC_FEATURES ========================
    # Mục đích: Từ obs của Actor tạo tensor [batch, số_cạnh, 4]; bỏ ego/padded, đếm luồng cùng/ ngược và áp lực next_node.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def traffic_features(self, obs: dict) -> torch.Tensor:
        """[batch, E, 4], exact order of self.edge_index, no stateful cache.

        The first AGV row is the current ego, excluded from traffic counts.
        Other rows are randomized/padded by Tobias; contributions are symmetric
        with respect to that order and masked by in_system.
        """

        #========================= MỤC NHỎ - LẤY OBSERVATION ==========================
        # Đọc AGV tensor; chiều đầu là batch, chiều thứ hai là số slot AGV.
        agvs = obs['agvs']
        b, n_agvs, dim = agvs.shape
        if dim < 18:
            raise ValueError('Routing observation missing moving/last/next fields')
        n = self.node_xy.shape[0]
        count = self.edge_index.shape[1]
        rec, send = self.edge_index[0], self.edge_index[1]

        # Only in-system, non-ego AGVs. Padded rows must not create fake traffic.

        #========================== MỤC NHỎ - LỌC AGV HỢP LỆ ==========================
        # Chỉ tính các xe đang trong hệ thống; loại ego ở hàng đầu và padding.
        valid = agvs[:, :, 0] > 0.5
        valid = valid.clone()
        valid[:, 0] = False
        last = self._node_indices(agvs[:, :, 2:4])
        nxt = self._node_indices(agvs[:, :, 4:6])
        moving = valid & (agvs[:, :, 1] > 0.5) & (last != nxt)

        # Directional moving flow, counted per actual directed (last,next) pair.
        # N*N may look dense, but N~106 and scatter is far cheaper than B*E*AGV
        # pairwise comparison in the PPO full-rollout microbatch.

        #======================= MỤC NHỎ - ĐẾM XE THEO CẶP NODE =======================
        # Đánh số cặp last/next thành một chỉ mục để scatter_add hiệu quả.
        pair_index = last * n + nxt
        pair_flow = agvs.new_zeros((b, n * n))
        pair_flow.scatter_add_(1, pair_index, moving.to(agvs.dtype))
        same = pair_flow[:, send * n + rec]
        opposite = pair_flow[:, rec * n + send]

        # A physical queue length is NOT available in Tobias obs. This feature
        # is explicitly only next-node pressure (vehicles approaching / at dst).

        #======================== MỤC NHỎ - ÁP LỰC NODE ĐẦU RA ========================
        # Đếm xe khác có next_node ở đầu ra; đây KHÔNG phải queue length vật lý.
        pressure_per_node = agvs.new_zeros((b, n))
        pressure_per_node.scatter_add_(1, nxt, valid.to(agvs.dtype))
        pressure = pressure_per_node[:, rec]
        active_arc = self.edge_nonself.to(agvs.dtype).unsqueeze(0)

        #======================== MỤC NHỎ - GHÉP EDGE FEATURES ========================
        # Ghép [chiều dài, luồng cùng, luồng ngược, áp lực đầu ra] và chuẩn hóa trên 20 slot.
        features = torch.stack((
            self.edge_lengths_norm.to(agvs.dtype).unsqueeze(0).expand(b, count),
            same / NORMALIZATION_AGV_SLOTS,
            opposite / NORMALIZATION_AGV_SLOTS,
            pressure / NORMALIZATION_AGV_SLOTS,
        ), dim=-1)
        # Self loops are structural, not physical arcs. No traffic on self arcs.
        return features * active_arc.unsqueeze(-1)
    #=================== KẾT THÚC MODULE CON 02 - TRAFFIC_FEATURES ====================




    #============================= MODULE CON 03 - LOGITS =============================
    # Mục đích: Giữ node encoder/action head V15; chèn edge_attr vào hai lớp GATv2 và trả về 5 logits Routing.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def logits(self, obs: dict) -> torch.Tensor:
        # Exact V15 node embedding/scatter/action head; ONLY conv uses edge_attr.
        agvs, stations = obs['agvs'], obs['stat']
        if agvs.shape[-1] < 19 or stations.shape[-1] < 2:
            raise ValueError('Incompatible Tobias AGV/station observation')
        b, a, _ = agvs.shape
        n = self.node_xy.shape[0]
        ego = agvs[:, :1]
        goal_input = torch.cat((ego[:, :, 4:8], ego[:, :, 18:19]), -1)
        goal_embed = self.goal_encoder(self.fourier(goal_input, [0, 2])).squeeze(1)
        nodes = self._node_indices(agvs[:, :, 2:16].reshape(b, a, 7, 2))
        indices = torch.cat((nodes, nodes[:, :, :1].expand(-1, -1, 2), nodes[:, :, 1:2]), -1)
        source = torch.ones((b, a, 10), device=agvs.device, dtype=agvs.dtype)
        source[:, :, 7] = agvs[:, :, 1]
        source[:, :, 8:10] = agvs[:, :, 16:18]

        def aggregate(agv_value, agv_index):
            out = torch.zeros((b, n, 10), device=agvs.device, dtype=agvs.dtype)
            for k in range(10):
                out[:, :, k].scatter_add_(1, agv_index[:, :, k], agv_value[:, :, k])
            return out


        #======================== MỤC NHỎ - XÂY NODE FEATURES =========================
        # Scatter ego và các AGV khác lên nút; station + khoảng cách mục tiêu cũng được mã hóa.
        main = aggregate(source[:, :1], indices[:, :1])
        others = aggregate(source[:, 1:], indices[:, 1:])
        station_idx = self._node_indices(stations[:, :, :2])
        station_channel = self._scatter(torch.ones((b, station_idx.shape[1], 1),
                                                   device=agvs.device, dtype=agvs.dtype), station_idx, n)
        distance = torch.linalg.vector_norm(self.node_xy[None, :, :] - ego[:, :, 6:8],
                                             dim=-1).unsqueeze(-1)
        node_info = torch.cat((main, others, station_channel, distance), -1).detach()
        node = self.node_encoder(node_info)

        #==================== MỤC NHỎ - GHÉP THÔNG TIN GIAO THÔNG =====================
        # Tính edge_attr lại ở MỖI quyết định từ obs, không dùng history cache.
        edge_attr = self.traffic_features(obs)

        #======================== MỤC NHỎ - HAI LỚP EDGE-GATV2 ========================
        # Truyền thông điệp theo cạnh có hướng rồi chuẩn hóa GraphNorm.
        for conv, norm in zip(self.convs, self.norms):
            node = norm(conv(node, self.edge_index, edge_attr))
        node = torch.cat((node[:, :-1], torch.zeros_like(node[:, -1:])), dim=1)

        #======================= MỤC NHỎ - ACTION HEAD ROUTING ========================
        # Ghép embedding của bốn node kề với goal embedding để tạo năm logits.
        neighbor_idx = self._node_indices(ego[:, :, 8:16].reshape(b, 4, 2))
        neighbors = torch.gather(node, 1, neighbor_idx.unsqueeze(-1).expand(-1, -1, 32)).flatten(1)
        return self.action_net(torch.cat((neighbors, goal_embed), -1))
    #======================== KẾT THÚC MODULE CON 03 - LOGITS =========================




    #============================== MODULE CON 04 - DIST ==============================
    # Mục đích: Trả Categorical theo năm logits, chủ đích KHÔNG áp Routing action mask.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def dist(self, obs: dict) -> Categorical:
        # Intentional experimental parity: ROUTING UNMASKED.
        return Categorical(logits=self.logits(obs))
    #========================= KẾT THÚC MODULE CON 04 - DIST ==========================

#==================== KẾT THÚC MODULE 04 - LỚP TRAFFICROUTINGACTOR ====================




#============================ MODULE 05 - HÀM INSTALL_V16B ============================
# Dựng V15 trước, sau đó thay riêng Routing Actor bằng TrafficRoutingActor V16-B.
def install_v16b(model, obs_space, global_feature_dim, graph, v15_install):
    """Install original V15 architecture; replace only routing actor/optimizer."""
    arch = v15_install(model, obs_space, global_feature_dim, graph)
    old = model.roles['routing'].actor
    new = TrafficRoutingActor(graph).to(model.device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    if set(missing) != {'convs.0.edge_lin.weight', 'convs.1.edge_lin.weight'} or unexpected:
        raise RuntimeError(f'Unexpected V15 architecture change: {missing}, {unexpected}')
    model.roles['routing'].actor = new
    model.roles['routing'].actor_optimizer = torch.optim.Adam(new.parameters(), lr=model.cfg.actor_lr)
    return {**arch, 'routing': 'V16B_Traffic_Edge_GATv2_2x4_32_unmasked',
            'edge_features': list(EDGE_FEATURE_NAMES), 'traffic_from_actor_observation': True}
#======================= KẾT THÚC MODULE 05 - HÀM INSTALL_V16B ========================




#====================== MODULE 06 - HÀM TRANSFER_V15_CHECKPOINT =======================
# Warm-start actor/critic/ValueNorm từ checkpoint V15, giữ các trọng số tương thích.
def transfer_v15_checkpoint(model, checkpoint):
    """V15->V16-B conversion: preserve all compatible learned state.

    Routing's Adam is RESET because new edge parameters change its parameter
    groups. This is deliberate; DO NOT claim full optimizer transfer.
    Dispatch and all critics preserve optimizers and ValueNorm unchanged.
    A V16-B checkpoint is loaded strictly via MAPPO.load, including all Adam.
    """
    from pathlib import Path
    data = torch.load(str(Path(checkpoint).resolve()), map_location=model.device, weights_only=True)
    if set(data.get('roles', {})) != set(model.roles):
        raise RuntimeError('Checkpoint missing one of the 3 V15 roles')
    saved_routing = data['roles']['routing']['actor']
    traffic_keys = {'convs.0.edge_lin.weight', 'convs.1.edge_lin.weight'}
    present = traffic_keys.intersection(saved_routing)
    if present:
        if present != traffic_keys:
            raise RuntimeError('Partial V16-B edge parameters in checkpoint')
        model.load(str(Path(checkpoint).resolve()))
        return 'v16b_strict_resume_all_states'

    for name, net in model.roles.items():

        #========================= MỤC NHỎ - ĐỌC TRỌNG SỐ GỐC =========================
        # Lấy state_dict của V15 để warm-start các phần không thay đổi.
        saved = data['roles'][name]
        if name == 'routing':
            missing, unexpected = net.actor.load_state_dict(saved['actor'], strict=False)
            if set(missing) != traffic_keys or unexpected:
                raise RuntimeError(f'V15 Routing actor incompatible: {missing} {unexpected}')
            # Edge weights zero, other routing weights exactly from V15.
            # Reset routing actor Adam; optimizer from V15 misses new tensors.
            net.actor_optimizer = torch.optim.Adam(net.actor.parameters(), lr=model.cfg.actor_lr)
        else:
            net.actor.load_state_dict(saved['actor'], strict=True)
            net.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        net.critic.load_state_dict(saved['critic'], strict=True)
        net.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        net.valuenorm.load_state_dict(saved['valuenorm'])
    return 'v15_compatible_weights_routing_actor_adam_reset'
#================== KẾT THÚC MODULE 06 - HÀM TRANSFER_V15_CHECKPOINT ==================


