"""V15: Tobias-inspired neural architectures on pure PyTorch.

Preserves the architectural pattern of Tobias's ma_gnn_routing.py and
ma_lin_net.py (GATv2 2 x 4 heads; 32-d embeddings; three separate LinFE MLP
branches; Fourier 4). This is an independent torch-only implementation, not a
byte-identical Torch-Geometric / RLlib weight-compatible copy. MAPPO itself is
unchanged. Three role-specific centralized critics see unshuffled physical state.

No Dijkstra, no dynamic traffic features, no extra reward shaping.
Routing intentionally uses NO action masking (Tobias joint-routing experiment).
Dispatch action masking remains enabled and unchanged.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Categorical


def _masked(logits, action_mask):
    # Exact math of thesis/mappo/networks.py:masked_logits (local copy for testability).
    # For Dispatch only; Routing uses the raw five logits without action masking.
    if logits.shape!=action_mask.shape:
        raise ValueError('Action/logit shape mismatch')
    valid=action_mask>.5
    if (~valid.any(-1)).any():
        raise ValueError('No legal actions')
    return logits.masked_fill(~valid,torch.finfo(logits.dtype).min/2)


class TobiasFourier(nn.Module):
    """Tobias's 4 Fourier frequencies per 2D point: sin/cos of 2*pi*B*x.
    Random B is frozen and included in checkpoints for exact resume.
    """
    def __init__(self, count=4):
        super().__init__()
        self.register_buffer('frequency', torch.randn(count, 2))

    def forward(self, x, columns):
        pos = torch.stack([x[..., c:c+2] for c in columns], dim=-2)
        radians = 2 * math.pi * torch.einsum('...pc,fc->...pf', pos, self.frequency)
        values = torch.stack((torch.sin(radians), torch.cos(radians)), -1)
        return torch.cat([x, values.flatten(-3)], dim=-1)


def _stack_mlp(in_dim, widths, bn_channels, activation):
    layers=[]
    last=in_dim
    for d in widths:
        layers.extend([nn.Linear(last,d), activation(), nn.BatchNorm1d(bn_channels)])
        last=d
    return nn.Sequential(*layers)


class TobiasLinearDispatchActor(nn.Module):
    """LinFE and 160->128->32->n action head from Tobias joint notebook.

    With n_AGVs=20, all physical AGV slots stay padded to the same size while
    actual fleet count changes. No trainable candidate scoring or attention.
    """
    uses_action_mask = True
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

    def dist(self, obs):
        return Categorical(logits=_masked(self.logits(obs),obs['action_mask']))


class TobiasGATv2(nn.Module):
    """Dense directed GATv2 with 4 heads and concat=False, no residual/edge data.

    A torch-only port of the GATv2 mathematical layer used by Tobias. On a
    105-node graph this avoids installing legacy PyG/torch_scatter/RLlib. The
    parameters are NOT interchangeable with torch_geometric.GATv2Conv.
    """
    def __init__(self, embedding=32, heads=4):
        super().__init__()
        self.embedding=embedding;self.heads=heads
        self.linear_src=nn.Linear(embedding, embedding*heads)
        self.linear_dst=nn.Linear(embedding, embedding*heads)
        self.att=nn.Parameter(torch.empty(heads,embedding))
        self.bias=nn.Parameter(torch.zeros(embedding))
        nn.init.xavier_uniform_(self.att)

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


class TobiasGraphNorm(nn.Module):
    """GraphNorm per sample, including PyG's trainable mean_scale."""
    def __init__(self, embedding=32):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(embedding))
        self.bias=nn.Parameter(torch.zeros(embedding))
        self.mean_scale=nn.Parameter(torch.ones(embedding))

    def forward(self,x):
        mean=x.mean(1,keepdim=True)
        centered=x-mean*self.mean_scale
        var=centered.square().mean(1,keepdim=True)
        return centered*(var+1e-5).rsqrt()*self.weight+self.bias


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

    def _node_indices(self, coords):
        # Padded rows are represented by (0,0) sentinel, exactly as Tobias.
        dif=(coords.unsqueeze(-2)-self.node_xy).abs().amax(-1)
        val,idx=dif.min(-1)
        if (val>1e-3).any():
            raise ValueError('Observation coordinates not present in Tobias Hive graph')
        return idx

    def _scatter(self,values,idx,number):
        b,m,c=values.shape
        ret=torch.zeros((b,number,c),device=values.device,dtype=values.dtype)
        return ret.scatter_add(1,idx.unsqueeze(-1).expand(-1,-1,c),values)

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

    def dist(self,obs):
        # Tobias routing experiment: no action mask on the network output.
        # Illegal directions remain sampleable and are handled by AnyLogic.
        # Keeping five logits preserves the V15 checkpoint architecture.
        return Categorical(logits=self.logits(obs))


class V15CentralCritic(nn.Module):
    """CTDE value network: full physical global state, per-role 160->128->128->32->1.

    The critic's 3 value-head hidden layers follow Tobias MLP widths. It is
    not Tobias's original local RLlib value function: it is intentionally
    centralized, consistent with MAPPO's global_state interface.
    """
    def __init__(self, feature_dim, embedding=32):
        super().__init__()
        if embedding!=32: raise ValueError('Expected embedding=32')
        self.agv=nn.Sequential(nn.Linear(feature_dim,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.station=nn.Sequential(nn.Linear(feature_dim,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
        self.value_head=nn.Sequential(nn.Linear(160,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),
                                      nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1))

    @staticmethod
    def _pool(v,mask):
        weight=mask.unsqueeze(-1)
        mean=(v*weight).sum(1)/weight.sum(1).clamp_min(1)
        maximum=v.masked_fill(~mask.unsqueeze(-1),torch.finfo(v.dtype).min/4).max(1).values
        maximum=torch.where(mask.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
        return mean,maximum

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


def install_v15(model, obs_space, global_feature_dim, graph):
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
