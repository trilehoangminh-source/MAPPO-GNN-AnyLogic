#=============== V17-B: 7 EDGE FEATURES + EDGE-CONDITIONED MESSAGE PASSING ===============
# Kế thừa 4 feature V16-B, bổ sung 3 feature, đưa edge vào Attention VÀ Message.
# Giữ nguyên 2 GAT layers / 4 heads / embedding32 / routing UNMASKED / Reward Tobias.
# V17-B PHẢI khởi tạo run mới; checkpoint V16-B không được resume bằng source V17-B.
#============================ HẾT GIỚI THIỆU FILE ============================


"""V17-B: 7 causal edge features; edge-conditioned attention AND message.

Authoritative design: THIET_KE_V17B_7_EDGE_FEATURES_EDGE_CONDITIONED_MESSAGE_PASSING.md
obs AGV[...,0]=in_system, 1=moving, 2:4=last_node, 4:6=next_node,
6:8=target_node; ego is slot 0 and is excluded from OTHER traffic counts.
All updates are recomputed from actor-visible observations; directed topology
is from Hive GraphSpec. Stationary occupancy is only a conservative proxy.
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
    'other_stationary_last_equals_next_proxy_per_20',
    'other_goal_destination_demand_per_20',
    'topological_bottleneck_score',
)
EDGE_DIM = len(EDGE_FEATURE_NAMES)
NORMALIZATION_AGV_SLOTS = 20
#=================== KẾT THÚC MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ===================




#============================ MODULE 03 - LỚP TRAFFICGATV2 ============================
# V17-B: edge_attr đi vào cả attention score và nội dung message.
class TrafficGATv2(TobiasGATv2):
    """Tobias GATv2 with independent edge attention/message projections.

    Both edge projections start at zero: transferred V15 weights initially
    reproduce node-only V15 GATv2 at the same fixed input.
    """



    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Hai phép chiếu cạnh độc lập (attention và message), khởi tạo 0 để tương thích V15.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, embedding: int = 32, heads: int = 4):
        super().__init__(embedding=embedding, heads=heads)
        self.edge_att_lin = nn.Linear(EDGE_DIM, embedding * heads, bias=False)
        self.edge_msg_lin = nn.Linear(EDGE_DIM, embedding * heads, bias=False)
        nn.init.zeros_(self.edge_att_lin.weight)
        nn.init.zeros_(self.edge_msg_lin.weight)
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================ MODULE CON 02 - FORWARD =============================
    # Mục đích: Tính attention và message có điều kiện cạnh, gom theo node nhận.
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
        edge_att = self.edge_att_lin(edge_attr).view(b, count, h, d)
        edge_msg = self.edge_msg_lin(edge_attr).view(b, count, h, d)

        #========================= MỤC NHỎ - ATTENTION SCORE ==========================
        # Embedding cạnh cho attention; message có phép chiếu riêng bên dưới.
        pair = nn.functional.leaky_relu(left[:, rec] + right[:, send] + edge_att, .2)
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
        # V17-B: nội dung message = node nguồn + thông tin cạnh (khác với V16-B).
        messages = sources + edge_msg.permute(0, 2, 1, 3)
        weighted = messages * weights.unsqueeze(-1)
        result = torch.zeros((b, h, n, d), dtype=x.dtype, device=x.device)
        result.scatter_add_(2, index.unsqueeze(-1).expand(-1, -1, -1, d), weighted)
        return result.mean(1) + self.bias
    #======================== KẾT THÚC MODULE CON 02 - FORWARD ========================

#======================= KẾT THÚC MODULE 03 - LỚP TRAFFICGATV2 ========================




#======================== MODULE 04 - LỚP TRAFFICROUTINGACTOR =========================
# Routing V17-B: giữ encoder V15, 7 feature cạnh, attention + message.
class TrafficRoutingActor(TobiasGraphRoutingActor):
    """Drop-in V15 actor with observation-only 7-edge message/attention."""
    action_dim = 5
    uses_action_mask = False



    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Thay GAT V15 bằng V17-B, nạp node weights, tính độ dài và bottleneck tĩnh.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, graph, embedding: int = 32, heads: int = 4, layers: int = 2):
        super().__init__(graph, embedding=embedding, heads=heads, layers=layers)
        old = self.convs
        self.convs = nn.ModuleList([TrafficGATv2(embedding, heads) for _ in range(layers)])
        for new, prev in zip(self.convs, old):
            missing, extra = new.load_state_dict(prev.state_dict(), strict=False)
            if set(missing) != {'edge_att_lin.weight', 'edge_msg_lin.weight'} or extra:
                raise RuntimeError(f'Bad V15 GAT transfer: missing={missing}, extra={extra}')
        src, dst = self.edge_index[1], self.edge_index[0]
        lengths = torch.linalg.vector_norm(self.node_xy[src] - self.node_xy[dst], dim=-1)
        nonself = src != dst
        real = lengths[nonself]
        max_len = real.max().clamp_min(1e-6) if real.numel() else lengths.new_tensor(1.)
        self.register_buffer('edge_lengths_norm', (lengths / max_len) * nonself.to(lengths.dtype), persistent=False)
        self.register_buffer('edge_nonself', nonself, persistent=False)

        # Feature 7: static directed exit restriction, excluding reverse to sender.
        # GraphSpec.arcs contains ONLY real directed arcs, not dummy/self loops.
        real_arcs = {(int(u), int(v)) for u, v in graph.arcs}
        out_neighbors: dict[int, set[int]] = {}
        for u, v in real_arcs:
            out_neighbors.setdefault(u, set()).add(v)
        bottleneck = lengths.new_zeros((len(src),))
        for edge_k, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            if u != v and (u, v) in real_arcs:
                exits = len(out_neighbors.get(v, set()) - {u})
                bottleneck[edge_k] = 1.0 / (1.0 + exits)
        self.register_buffer('edge_bottleneck', bottleneck, persistent=False)
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #======================== MODULE CON 02 - TRAFFIC_FEATURES ========================
    # Mục đích: 7 đặc trưng cạnh [B,E,7] lấy từ observation hiện tại, không dùng oracle.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def traffic_features(self, obs: dict) -> torch.Tensor:
        """Return [batch, E, 7] in the fixed V17-B feature order.

        Receiver is edge_index[0], sender is edge_index[1]. Exclude ego row 0,
        non-system and padded AGVs. f5 is a conservative stationary PROXY.
        """
        agvs = obs['agvs']
        b, n_agvs, dim = agvs.shape
        if dim < 18:
            raise ValueError('Routing observation missing 18 required AGV fields')
        n = self.node_xy.shape[0]
        count = self.edge_index.shape[1]
        rec, send = self.edge_index[0], self.edge_index[1]
        valid = (agvs[:, :, 0] > 0.5).clone()
        valid[:, 0] = False  # ignore ego AGV
        last = self._node_indices(agvs[:, :, 2:4])
        nxt = self._node_indices(agvs[:, :, 4:6])
        goal = self._node_indices(agvs[:, :, 6:8])
        moving = valid & (agvs[:, :, 1] > 0.5) & (last != nxt)

        # f2/f3: actual directional (last, next) pairs for OTHER moving AGVs.
        pair_index = last * n + nxt
        pair_flow = agvs.new_zeros((b, n * n))
        pair_flow.scatter_add_(1, pair_index, moving.to(agvs.dtype))
        same = pair_flow[:, send * n + rec]
        opposite = pair_flow[:, rec * n + send]

        # f4: next-node pressure; f5: reliable stationary proxy (last==next);
        # f6: target-node demand. These node-centric counts share receiver v.
        pressure_node = agvs.new_zeros((b, n))
        pressure_node.scatter_add_(1, nxt, valid.to(agvs.dtype))
        pressure = pressure_node[:, rec]

        stationary = valid & (agvs[:, :, 1] <= 0.5) & (last == nxt)
        stationary_node = agvs.new_zeros((b, n))
        stationary_node.scatter_add_(1, nxt, stationary.to(agvs.dtype))
        occupancy_proxy = stationary_node[:, rec]

        demand_node = agvs.new_zeros((b, n))
        demand_node.scatter_add_(1, goal, valid.to(agvs.dtype))
        demand = demand_node[:, rec]

        # f1 and f7 are static from the REAL graph; f7 ignores self/dummy.
        static_length = self.edge_lengths_norm.to(dtype=agvs.dtype).unsqueeze(0).expand(b, count)
        static_bottleneck = self.edge_bottleneck.to(dtype=agvs.dtype).unsqueeze(0).expand(b, count)
        features = torch.stack((
            static_length,
            same / NORMALIZATION_AGV_SLOTS,
            opposite / NORMALIZATION_AGV_SLOTS,
            pressure / NORMALIZATION_AGV_SLOTS,
            occupancy_proxy / NORMALIZATION_AGV_SLOTS,
            demand / NORMALIZATION_AGV_SLOTS,
            static_bottleneck,
        ), dim=-1)
        active_arc = self.edge_nonself.to(dtype=agvs.dtype).view(1, count, 1)
        return features * active_arc
    #=================== KẾT THÚC MODULE CON 02 - TRAFFIC_FEATURES ====================




    #============================= MODULE CON 03 - LOGITS =============================
    # Mục đích: Giữ node encoder/action head V15, chèn edge_attr vào attention và message ở cả hai lớp.
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
        # 7 edge features tính ở mỗi quyết định, không dùng history cache/oracle.
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
# Dựng V15 trước, sau đó thay riêng Routing Actor bằng TrafficRoutingActor V17-B.
def install_v16b(model, obs_space, global_feature_dim, graph, v15_install):
    """Install original V15 architecture; replace only routing actor/optimizer."""
    arch = v15_install(model, obs_space, global_feature_dim, graph)
    old = model.roles['routing'].actor
    new = TrafficRoutingActor(graph).to(model.device)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    v17_new = {f'convs.{k}.{key}.weight'
               for k in range(2) for key in ('edge_att_lin', 'edge_msg_lin')}
    if set(missing) != v17_new or unexpected:
        raise RuntimeError(f'Unexpected V15 architecture change: {missing}, {unexpected}')
    model.roles['routing'].actor = new
    model.roles['routing'].actor_optimizer = torch.optim.Adam(new.parameters(), lr=model.cfg.actor_lr)
    return {**arch, 'routing': 'V17B_Edge7_Conditioned_Message_GATv2_2x4_32_unmasked',
             'edge_features': list(EDGE_FEATURE_NAMES),
             'edge_attention_conditioned': True,
             'edge_message_conditioned': True,
             'traffic_from_actor_observation': True}
#======================= KẾT THÚC MODULE 05 - HÀM INSTALL_V16B ========================




#====================== MODULE 06 - HÀM TRANSFER_V15_CHECKPOINT =======================
# V17-B warm-start từ V15 Joint16 EP0, giữ trọng số tương thích, reset Adam Routing.
def transfer_v15_checkpoint(model, checkpoint):
    """V15 Joint16 EP0 -> V17-B (or exact V17-B strict resume).

    Actor V15 parameters transferred strictly except 2x(att, message) new
    edge weights. Routing Actor Adam is reset due to parameter-group changes.
    Dispatch/critics/ValueNorm and their compatible optimizers are preserved.
    """
    from pathlib import Path
    data = torch.load(str(Path(checkpoint).resolve()), map_location=model.device, weights_only=True)
    if set(data.get('roles', {})) != set(model.roles):
        raise RuntimeError('Checkpoint missing a V15/V17-B role')
    saved_routing = data['roles']['routing']['actor']
    keys = {f'convs.{k}.{kind}.weight' for k in range(2)
            for kind in ('edge_att_lin', 'edge_msg_lin')}
    if keys.issubset(saved_routing):
        # Native V17-B: strict reload is required for trustworthy resume.
        model.load(str(Path(checkpoint).resolve()))
        return 'v17b_strict_resume_all_states'
    if keys.intersection(saved_routing) or any('edge_lin.weight' in k for k in saved_routing):
        raise RuntimeError('Unsupported partial/V16-B edge checkpoint: use V15 Joint16 EP0')

    for name, net in model.roles.items():
        saved = data['roles'][name]
        if name == 'routing':
            missing, unexpected = net.actor.load_state_dict(saved['actor'], strict=False)
            if set(missing) != keys or unexpected:
                raise RuntimeError(f'V15 Routing checkpoint incompatible: {missing} {unexpected}')
            net.actor_optimizer = torch.optim.Adam(net.actor.parameters(), lr=model.cfg.actor_lr)
        else:
            net.actor.load_state_dict(saved['actor'], strict=True)
            net.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        net.critic.load_state_dict(saved['critic'], strict=True)
        net.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        net.valuenorm.load_state_dict(saved['valuenorm'])
    return 'v15_compatible_weights_routing_actor_adam_reset'
#================== KẾT THÚC MODULE 06 - HÀM TRANSFER_V15_CHECKPOINT ==================


