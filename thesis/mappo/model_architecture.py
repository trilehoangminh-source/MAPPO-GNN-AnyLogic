#======= KIẾN TRÚC MAPPO: ROUTING EDGE-GATV2, DISPATCH LINFE VÀ CTDE CRITIC =======
# Một module chứa kiến trúc nền, Routing dùng bảy edge features và chuyển checkpoint.
#================================ HẾT GIỚI THIỆU FILE =================================


"""MAPPO neural architectures on pure PyTorch.

Preserves the architectural pattern of Tobias's ma_gnn_routing.py and
ma_lin_net.py (GATv2 2 x 4 heads; 32-d embeddings; three separate LinFE MLP
branches; Fourier 4). This is an independent torch-only implementation, not a
byte-identical Torch-Geometric / RLlib weight-compatible copy. MAPPO itself is
unchanged. Three role-specific centralized critics see unshuffled physical state.

The active TrafficRoutingActor adds seven observation-derived edge features to
attention and message passing. Routing intentionally uses no action mask.
Dispatch action masking remains enabled and unchanged.
"""



#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Categorical
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================




#============================== MODULE 02 - HÀM _MASKED ===============================
# Áp mặt nạ hợp lệ vào vector logits cho Dispatch.
def _masked(logits, action_mask):
    # Exact math of thesis/mappo/networks.py:masked_logits (local copy for testability).
    # For Dispatch only; Routing uses the raw five logits without action masking.
    if logits.shape!=action_mask.shape:
        raise ValueError('Action/logit shape mismatch')
    valid=action_mask>.5
    if (~valid.any(-1)).any():
        raise ValueError('No legal actions')
    return logits.masked_fill(~valid,torch.finfo(logits.dtype).min/2)
#========================== KẾT THÚC MODULE 02 - HÀM _MASKED ==========================




#=========================== MODULE 03 - LỚP TOBIASFOURIER ============================
# Mã hóa tọa độ/mục tiêu thành Fourier features.
class TobiasFourier(nn.Module):
    """Tobias's 4 Fourier frequencies per 2D point: sin/cos of 2*pi*B*x.
    Random B is frozen and included in checkpoints for exact resume.
    """


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, count=4):
        super().__init__()
        self.register_buffer('frequency', torch.randn(count, 2))
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================ MODULE CON 02 - FORWARD =============================
    # Mục đích: Tính phép biến đổi tensor theo luồng đi tới của mạng PyTorch.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def forward(self, x, columns):
        pos = torch.stack([x[..., c:c+2] for c in columns], dim=-2)
        radians = 2 * math.pi * torch.einsum('...pc,fc->...pf', pos, self.frequency)
        values = torch.stack((torch.sin(radians), torch.cos(radians)), -1)
        return torch.cat([x, values.flatten(-3)], dim=-1)
    #======================== KẾT THÚC MODULE CON 02 - FORWARD ========================

#======================= KẾT THÚC MODULE 03 - LỚP TOBIASFOURIER =======================




#============================= MODULE 04 - HÀM _STACK_MLP =============================
# Dựng MLP bằng danh sách kích thước lớp và activation.
def _stack_mlp(in_dim, widths, bn_channels, activation):
    layers=[]
    last=in_dim
    for d in widths:
        layers.extend([nn.Linear(last,d), activation(), nn.BatchNorm1d(bn_channels)])
        last=d
    return nn.Sequential(*layers)
#======================== KẾT THÚC MODULE 04 - HÀM _STACK_MLP =========================




#===================== MODULE 05 - LỚP TOBIASLINEARDISPATCHACTOR ======================
# Actor LinFE phục vụ Dispatch Front/Rear.
class TobiasLinearDispatchActor(nn.Module):
    """LinFE and 160->128->32->n action head from Tobias joint notebook.

    With n_AGVs=20, all physical AGV slots stay padded to the same size while
    actual fleet count changes. No trainable candidate scoring or attention.
    """
    uses_action_mask = True


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, obs_space, n_actions, embedding=32):
        super().__init__()
        agv_slots=int(obs_space['agvs'].shape[0])
        stat_slots=int(obs_space['stat'].shape[0])
        feature_dim=int(obs_space['agvs'].shape[-1])
        if agv_slots<2 or stat_slots<1 or embedding!=32:
            raise ValueError('V15 expects at least 2 padded AGV slots, stations and embed=32')
        self.action_dim=int(n_actions)
        self.agv_slots=agv_slots
        self.station_slots=stat_slots
        self.fourier=TobiasFourier(4)
        # Seven 2-D coordinate pairs in AGV rows; one pair in station rows.
        d_agv=feature_dim + 7*8
        d_stat=feature_dim + 8
        self.main_ff=_stack_mlp(d_agv,[64,64,64,32],1,nn.ReLU)
        self.other_ff=_stack_mlp(2*d_agv,[128,128,64,32],agv_slots-1,nn.ReLU)
        self.station_ff=_stack_mlp(d_agv+d_stat,[128,128,64,32],stat_slots,nn.ReLU)
        self.action_net=nn.Sequential(nn.Linear(160,128),nn.ReLU(),
                                      nn.Linear(128,32),nn.ReLU(),
                                      nn.Linear(32,self.action_dim))
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================= MODULE CON 02 - LOGITS =============================
    # Mục đích: Tính vector điểm chưa chuẩn hóa cho các hành động khả thi.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def logits(self, obs):
        agv=obs['agvs'];stations=obs['stat']
        b,n,_=agv.shape
        if n!=self.agv_slots or stations.shape[1]!=self.station_slots:
            raise ValueError('AGV/station padding changed: cannot safely reuse checkpoint')
        agv_fourier=self.fourier(agv,[2,4,6,8,10,12,14])
        stat_fourier=self.fourier(stations,[0])
        ego=agv_fourier[:,:1]
        ego_embed=self.main_ff(ego).squeeze(1)
        others=agv_fourier[:,1:]
        other_embedding=self.other_ff(torch.cat([ego.expand(-1,n-1,-1),others],-1))
        stations_embedding=self.station_ff(torch.cat([
            ego.expand(-1,stations.shape[1],-1),stat_fourier],-1))
        features=torch.cat((ego_embed,other_embedding.mean(1),other_embedding.max(1).values,
                            stations_embedding.max(1).values,stations_embedding.mean(1)),-1)
        return self.action_net(features)
    #======================== KẾT THÚC MODULE CON 02 - LOGITS =========================




    #============================== MODULE CON 03 - DIST ==============================
    # Mục đích: Tạo phân phối xác suất hành động Categorical từ logits của Actor.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def dist(self, obs):
        return Categorical(logits=_masked(self.logits(obs),obs['action_mask']))
    #========================= KẾT THÚC MODULE CON 03 - DIST ==========================

#================= KẾT THÚC MODULE 05 - LỚP TOBIASLINEARDISPATCHACTOR =================




#============================ MODULE 06 - LỚP TOBIASGATV2 =============================
# Lớp attention trên graph V15 không có thuộc tính cạnh động.
class TobiasGATv2(nn.Module):
    """Dense directed GATv2 with 4 heads and concat=False, no residual/edge data.

    A torch-only port of the GATv2 mathematical layer used by Tobias. On a
    105-node graph this avoids installing legacy PyG/torch_scatter/RLlib. The
    parameters are NOT interchangeable with torch_geometric.GATv2Conv.
    """


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, embedding=32, heads=4):
        super().__init__()
        self.embedding=embedding;self.heads=heads
        self.linear_src=nn.Linear(embedding, embedding*heads)
        self.linear_dst=nn.Linear(embedding, embedding*heads)
        self.att=nn.Parameter(torch.empty(heads,embedding))
        self.bias=nn.Parameter(torch.zeros(embedding))
        nn.init.xavier_uniform_(self.att)
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================ MODULE CON 02 - FORWARD =============================
    # Mục đích: Tự cài attention: chiếu node, tính attention score, softmax theo receiver và gom thông điệp.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def forward(self, x, edge_index):
        # x[B,N,D]; edge_index[2,E] has receiver row and sender row.
        # Sparse aggregation avoids O(B*heads*N*N*dim) dense memory.
        b,n,d=x.shape;h=self.heads
        rec,send=edge_index[0],edge_index[1]
        count=rec.numel()
        left=self.linear_dst(x).view(b,n,h,d)
        right=self.linear_src(x).view(b,n,h,d)
        pair=nn.functional.leaky_relu(left[:,rec]+right[:,send],.2)
        scores=(pair*self.att[None,None]).sum(-1).permute(0,2,1) # [B,H,E]
        index=rec.view(1,1,count).expand(b,h,-1)
        # Stable edge-wise softmax over each receiver's incoming neighbors.
        # The detached maximum is a numerical shift; normalizer gradient is exact.
        with torch.no_grad():
            maximum=torch.full((b,h,n),-float('inf'),dtype=scores.dtype,device=x.device)
            maximum.scatter_reduce_(2,index,scores.detach(),reduce='amax',include_self=True)
        weights=torch.exp(scores-maximum.gather(2,index))
        sums=torch.zeros((b,h,n),dtype=weights.dtype,device=x.device)
        sums.scatter_add_(2,index,weights)
        weights=weights/sums.gather(2,index).clamp_min(1e-12)
        sources=right[:,send].permute(0,2,1,3)   # [B,H,E,D]
        weighted=sources*weights.unsqueeze(-1)
        result=torch.zeros((b,h,n,d),dtype=x.dtype,device=x.device)
        result.scatter_add_(2,index.unsqueeze(-1).expand(-1,-1,-1,d),weighted)
        return result.mean(1)+self.bias
    #======================== KẾT THÚC MODULE CON 02 - FORWARD ========================

#======================== KẾT THÚC MODULE 06 - LỚP TOBIASGATV2 ========================




#========================== MODULE 07 - LỚP TOBIASGRAPHNORM ===========================
# Chuẩn hóa node embedding giữa các lớp GAT.
class TobiasGraphNorm(nn.Module):
    """GraphNorm per sample, including PyG's trainable mean_scale."""


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, embedding=32):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(embedding))
        self.bias=nn.Parameter(torch.zeros(embedding))
        self.mean_scale=nn.Parameter(torch.ones(embedding))
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================ MODULE CON 02 - FORWARD =============================
    # Mục đích: Tính phép biến đổi tensor theo luồng đi tới của mạng PyTorch.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def forward(self,x):
        mean=x.mean(1,keepdim=True)
        centered=x-mean*self.mean_scale
        var=centered.square().mean(1,keepdim=True)
        return centered*(var+1e-5).rsqrt()*self.weight+self.bias
    #======================== KẾT THÚC MODULE CON 02 - FORWARD ========================

#====================== KẾT THÚC MODULE 07 - LỚP TOBIASGRAPHNORM ======================




#====================== MODULE 08 - LỚP TOBIASGRAPHROUTINGACTOR =======================
# Actor V15: 22 node features, 2 GATv2, 4 heads, 32 chiều, 5 action, không mask.
class TobiasGraphRoutingActor(nn.Module):
    """Two GATv2Conv->GraphNorm layers, 4 heads, embedding=32, Tobias action head.

    Node feature construction is based on ma_gnn_routing.GNNFeatureExtractor:
    - AGV coordinate-to-graph scatter (main 10 channels, others 10),
    - station scatter (1), target distance (1), total 22 per node,
    - 4 neighboring node embeddings + ego goal embedding => 5x32.
    Routing action dimension is 5 for MAPPO (Stop + four directions), not the
    6-slot padded RLlib output used in Tobias's GNNRoutingNet.
    """
    action_dim=5
    uses_action_mask = False


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, graph, embedding=32, heads=4, layers=2):
        super().__init__()
        if (embedding,heads,layers)!=(32,4,2):
            raise ValueError('V15 fixes Tobias joint setup to embedding32, 4heads, 2GAT')
        xy=torch.tensor(list(graph.xy)+[(0.,0.)],dtype=torch.float32)
        count=len(xy)
        adj=torch.eye(count,dtype=torch.bool)
        for src,dst in graph.arcs: adj[int(dst),int(src)]=True
        self.register_buffer('node_xy',xy)
        receivers,senders=adj.nonzero(as_tuple=True)
        self.register_buffer('edge_index',torch.stack((receivers,senders),dim=0))
        self.register_buffer('graph_arcs',torch.as_tensor(graph.arcs,dtype=torch.long))
        self.graph_fingerprint=graph.fingerprint
        self.fourier=TobiasFourier(4)
        self.goal_encoder=_stack_mlp(21,[64,32],1,nn.LeakyReLU)
        self.node_encoder=_stack_mlp(22,[64,32],count,nn.LeakyReLU)
        self.convs=nn.ModuleList([TobiasGATv2(32,4) for _ in range(2)])
        self.norms=nn.ModuleList([TobiasGraphNorm(32) for _ in range(2)])
        self.action_net=nn.Sequential(nn.Linear(160,128),nn.LeakyReLU(),
                                      nn.Linear(128,32),nn.LeakyReLU(),nn.Linear(32,5))
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #========================= MODULE CON 02 - _NODE_INDICES ==========================
    # Mục đích: Hàm nội bộ node indices; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _node_indices(self, coords):
        # Padded rows are represented by (0,0) sentinel, exactly as Tobias.
        dif=(coords.unsqueeze(-2)-self.node_xy).abs().amax(-1)
        val,idx=dif.min(-1)
        if (val>1e-3).any():
            raise ValueError('Observation coordinates not present in Tobias Hive graph')
        return idx
    #===================== KẾT THÚC MODULE CON 02 - _NODE_INDICES =====================




    #============================ MODULE CON 03 - _SCATTER ============================
    # Mục đích: Đưa thông tin AGV vào node graph tương ứng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _scatter(self,values,idx,number):
        b,m,c=values.shape
        ret=torch.zeros((b,number,c),device=values.device,dtype=values.dtype)
        return ret.scatter_add(1,idx.unsqueeze(-1).expand(-1,-1,c),values)
    #======================= KẾT THÚC MODULE CON 03 - _SCATTER ========================




    #============================= MODULE CON 04 - LOGITS =============================
    # Mục đích: Tạo 22 kênh node features từ các AGV/station/goal; chạy GAT và nối 4 node ứng viên vào action head.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def logits(self, obs):
        agvs,stations=obs['agvs'],obs['stat']
        if agvs.shape[-1]<19 or stations.shape[-1]<2:
            raise ValueError('Incompatible Tobias AGV/station observation')
        b,a,_=agvs.shape;n=self.node_xy.shape[0]
        ego=agvs[:,:1]
        goal_input=torch.cat((ego[:,:,4:8],ego[:,:,18:19]),-1)
        goal_embed=self.goal_encoder(self.fourier(goal_input,[0,2])).squeeze(1)
        nodes=self._node_indices(agvs[:,:,2:16].reshape(b,a,7,2))
        indices=torch.cat((nodes,nodes[:,:,:1].expand(-1,-1,2),nodes[:,:,1:2]),-1)
        source=torch.ones((b,a,10),device=agvs.device,dtype=agvs.dtype)
        source[:,:,7]=agvs[:,:,1]
        source[:,:,8:10]=agvs[:,:,16:18]
        # Each of the ten coordinate roles contributes a 1D channel per node.
        def aggregate(agv_value,agv_index):
            out=torch.zeros((b,n,10),device=agvs.device,dtype=agvs.dtype)
            for k in range(10):
                out[:,:,k].scatter_add_(1,agv_index[:,:,k],agv_value[:,:,k])
            return out
        main=aggregate(source[:,:1],indices[:,:1])
        others=aggregate(source[:,1:],indices[:,1:])
        station_idx=self._node_indices(stations[:,:,:2])
        station_channel=self._scatter(torch.ones((b,station_idx.shape[1],1),device=agvs.device,dtype=agvs.dtype),station_idx,n)
        distance=torch.linalg.vector_norm(self.node_xy[None,:,:]-ego[:,:,6:8],dim=-1).unsqueeze(-1)
        node_info=torch.cat((main,others,station_channel,distance),-1).detach()
        node=self.node_encoder(node_info)
        for conv,norm in zip(self.convs,self.norms):
            node=norm(conv(node,self.edge_index))
        # Dummy node cannot become a legitimate Routing candidate.
        node=torch.cat((node[:,:-1],torch.zeros_like(node[:,-1:])),dim=1)
        neighbor_idx=self._node_indices(ego[:,:,8:16].reshape(b,4,2))
        neighbors=torch.gather(node,1,neighbor_idx.unsqueeze(-1).expand(-1,-1,32)).flatten(1)
        return self.action_net(torch.cat((neighbors,goal_embed),-1))
    #======================== KẾT THÚC MODULE CON 04 - LOGITS =========================




    #============================== MODULE CON 05 - DIST ==============================
    # Mục đích: Phân phối 5 hành động không mask để giữ thí nghiệm giống Tobias.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def dist(self,obs):
        # Tobias routing experiment: no action mask on the network output.
        # Illegal directions remain sampleable and are handled by AnyLogic.
        # Keeping five logits preserves the V15 checkpoint architecture.
        return Categorical(logits=self.logits(obs))
    #========================= KẾT THÚC MODULE CON 05 - DIST ==========================

#================== KẾT THÚC MODULE 08 - LỚP TOBIASGRAPHROUTINGACTOR ==================




#========================== MODULE 09 - LỚP V15CENTRALCRITIC ==========================
# Critic tập trung MLP trên AGV + station + ego state.
class V15CentralCritic(nn.Module):
    """CTDE value network: full physical global state, per-role 160->128->128->32->1.

    The critic's 3 value-head hidden layers follow Tobias MLP widths. It is
    not Tobias's original local RLlib value function: it is intentionally
    centralized, consistent with MAPPO's global_state interface.
    """


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, feature_dim, embedding=32):
        super().__init__()
        if embedding!=32: raise ValueError('Expected embedding=32')
        self.agv=nn.Sequential(nn.Linear(feature_dim,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.station=nn.Sequential(nn.Linear(feature_dim,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.value_head=nn.Sequential(nn.Linear(160,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),
                                      nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1))
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================= MODULE CON 02 - _POOL ==============================
    # Mục đích: Gộp đặc trưng nhiều thực thể thành vector cố định.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @staticmethod
    def _pool(v,mask):
        weight=mask.unsqueeze(-1)
        mean=(v*weight).sum(1)/weight.sum(1).clamp_min(1)
        maximum=v.masked_fill(~mask.unsqueeze(-1),torch.finfo(v.dtype).min/4).max(1).values
        maximum=torch.where(mask.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
        return mean,maximum
    #========================= KẾT THÚC MODULE CON 02 - _POOL =========================




    #============================ MODULE CON 03 - FORWARD =============================
    # Mục đích: Xử lý global AGV/station state và ego để xuất một giá trị V(s) cho Critic.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def forward(self,state):
        a,s,idx=state['agvs'],state['stat'],state['ego']
        b,n,_=a.shape
        if idx.shape!=(b,) or (idx<0).any() or (idx>=n).any():
            raise ValueError('Invalid physical AGV index for centralized critic')
        ae=self.agv(a);se=self.station(s)
        rows=torch.arange(b,device=a.device)
        ego=ae[rows,idx]
        other=(a.abs().sum(-1)>0).clone()
        other[rows,idx]=False
        am,ax=self._pool(ae,other)
        sm,sx=self._pool(se,s.abs().sum(-1)>0)
        return self.value_head(torch.cat((ego,am,ax,sm,sx),-1)).squeeze(-1)
    #======================== KẾT THÚC MODULE CON 03 - FORWARD ========================

#===================== KẾT THÚC MODULE 09 - LỚP V15CENTRALCRITIC ======================




#============================ MODULE 10 - HÀM INSTALL_V15 =============================
# Thay actor/critic placeholder thành mạng Tobias V15 và tạo optimizer tương ứng.
def install_base_architecture(model, obs_space, global_feature_dim, graph):
    """Install source-pattern architectures BEFORE EP0, rebuilding Adam from scratch."""
    required={'routing','dispatch_front','dispatch_rear'}
    if set(model.roles)!=required:
        raise ValueError('V15 requires routing, dispatch_front, dispatch_rear')
    for name,role in model.roles.items():
        if name=='routing':
            role.actor=TobiasGraphRoutingActor(graph).to(model.device)
        else:
            role.actor=TobiasLinearDispatchActor(obs_space,role.actor.action_dim).to(model.device)
        role.critic=V15CentralCritic(global_feature_dim).to(model.device)
        role.actor_optimizer=torch.optim.Adam(role.actor.parameters(),lr=model.cfg.actor_lr)
        role.critic_optimizer=torch.optim.Adam(role.critic.parameters(),lr=model.cfg.critic_lr)
        from thesis.mappo.algorithm import RunningValueNorm
        role.valuenorm=RunningValueNorm()
    return dict(routing='Tobias_GATv2_torch_port_2_layers_4heads_32',
                dispatch='Tobias_LinFE_torch_port_32',
                critics='per_role_CentralizedMLP_160_128_128_32_1',
                original_libraries_needed=False,torch_geometric_weights_compatible=False,
                dijkstra=False)
#======================== KẾT THÚC MODULE 10 - HÀM INSTALL_V15 ========================

# Traffic-aware routing actor and checkpoint conversion.
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
def install_traffic_routing(model, obs_space, global_feature_dim, graph, install_base):
    """Install original V15 architecture; replace only routing actor/optimizer."""
    arch = install_base(model, obs_space, global_feature_dim, graph)
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
def transfer_checkpoint(model, checkpoint):
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


