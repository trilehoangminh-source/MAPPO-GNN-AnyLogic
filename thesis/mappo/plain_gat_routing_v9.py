"""Vanilla node-only directed GAT for the existing Tobias MAPPO Routing actor.

No residual/skip connection and no edge-feature attention. Dispatching remains
its original MLP. Compatible with the B3 graph, Dijkstra prior and V7 PPO.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class PlainDirectedGAT(nn.Module):
    """Multi-head additive GAT on outgoing directed arcs plus self-loops.

    Input/output: [batch, graph_nodes, hidden]. No x + update skip path.
    """

    def __init__(self, hidden: int, heads: int = 2):
        super().__init__()
        if hidden < 4 or heads < 1 or hidden % heads:
            raise ValueError("Require hidden>=4, heads>=1, hidden % heads == 0")
        self.hidden, self.heads = int(hidden), int(heads)
        self.head_dim = hidden // heads
        self.linear = nn.Linear(hidden, hidden, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.negative_slope = 0.2
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, node: torch.Tensor, outgoing_average: torch.Tensor):
        if node.ndim != 3 or outgoing_average.ndim != 2:
            raise ValueError("Expect [B,N,H] nodes, [N,N] adjacency")
        batch, n, h = node.shape
        if h != self.hidden or outgoing_average.shape != (n, n):
            raise ValueError("GAT dimensions do not match graph")
        if not torch.isfinite(node).all():
            raise ValueError("Nonfinite GAT input")

        z = self.linear(node).reshape(batch, n, self.heads, self.head_dim)
        z = z.transpose(1, 2)  # [B,heads,N,head_dim]
        left = (z * self.att_src[None, :, None, :]).sum(-1)
        right = (z * self.att_dst[None, :, None, :]).sum(-1)
        logits = torch.nn.functional.leaky_relu(
            left.unsqueeze(-1) + right.unsqueeze(-2), self.negative_slope
        )  # [B,heads,source,target]
        allowed = outgoing_average.to(node.device) > 0
        allowed = allowed | torch.eye(n, device=node.device, dtype=torch.bool)
        logits = logits.masked_fill(
            ~allowed[None, None, :, :], torch.finfo(logits.dtype).min
        )
        attention = torch.softmax(logits, dim=-1)
        mixed = (attention @ z).transpose(1, 2).contiguous()
        mixed = mixed.reshape(batch, n, self.hidden)
        # Deliberately NO node + mixed or other residual/skip connection.
        return torch.tanh(self.norm(self.out(mixed)))


def enable_plain_gat_routing(model, heads: int = 2):
    """Replace only Routing message-passing layers, rebuild its optimizer."""
    from thesis.mappo.gnn_routing_B3_v121 import GraphRoutingActor

    role = model.roles.get("routing")
    if role is None or not isinstance(role.actor, GraphRoutingActor):
        raise RuntimeError("Expected B3 GraphRoutingActor in Routing role")
    if getattr(role.actor, "_plain_gat_v9", False):
        raise RuntimeError("Plain GAT already installed")
    if any(isinstance(v.actor, GraphRoutingActor)
           for name, v in model.roles.items() if name != "routing"):
        raise RuntimeError("Dispatch actor unexpectedly uses GNN")

    actor = role.actor
    layers = len(actor.layers)
    actor.layers = nn.ModuleList(
        PlainDirectedGAT(actor.hidden, heads) for _ in range(layers)
    ).to(model.device)
    role.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=model.cfg.actor_lr)
    actor._plain_gat_v9 = True
    return {
        "routing_encoder": "plain_directed_gat_no_residual_no_edge_features",
        "graph_layers": layers,
        "heads": heads,
        "residual": False,
        "edge_features": False,
        "directed_arcs": int((actor.outgoing_average > 0).sum().item()),
    }
