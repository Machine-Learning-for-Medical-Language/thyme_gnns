"""Heterogeneous Graph Transformer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import torch
import torch.nn as nn

from dgl import function as fn
from dgl.nn.pytorch.linear import TypedLinear
from dgl.nn.pytorch.softmax import edge_softmax

from dgl.ops import gather_mm, segment_mm

import pdb


class HGTConv(nn.Module):
    def __init__(
        self,
        in_size,
        head_size,
        num_heads,
        num_ntypes,
        num_etypes,
        dropout=0.2,
        use_norm=False,
    ):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm

        self.pre_reduction = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(
            head_size * num_heads, head_size * num_heads, num_ntypes
        )

        self.relation_pri = nn.ParameterList(
            [nn.Parameter(torch.ones(num_etypes)) for i in range(num_heads)]
        )
        self.relation_att = nn.ModuleList(
            [TypedLinear(head_size, head_size, num_etypes) for i in range(num_heads)]
        )
        self.relation_msg = nn.ModuleList(
            [TypedLinear(head_size, head_size, num_etypes) for i in range(num_heads)]
        )
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norm = nn.LayerNorm(head_size * num_heads)
        if in_size != head_size * num_heads:
            self.residual_w = nn.Parameter(torch.Tensor(in_size, head_size * num_heads))
            nn.init.xavier_uniform_(self.residual_w)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        self.presorted = presorted
        if g.is_block:
            x_src = x
            x_dst = x[: g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[: g.num_dst_nodes()]
        else:
            x_src = x
            x_dst = x
            srcntype = ntype
            dstntype = ntype
        with g.local_scope():
            k = self.linear_k(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            q = self.linear_q(x_dst, dstntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            v = self.linear_v(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            g.srcdata["k"] = k
            g.dstdata["q"] = q
            g.srcdata["v"] = v
            g.edata["etype"] = etype
            g.apply_edges(self.message)
            g.edata["m"] = g.edata["m"] * edge_softmax(g, g.edata["a"]).unsqueeze(-1)
            g.update_all(fn.copy_e("m", "m"), self.reduce_sum)
            h = g.dstdata["h"].view(-1, self.num_heads * self.head_size)
            # target-specific aggregation
            h = self.drop(self.linear_a(h, dstntype, presorted))
            alpha = torch.sigmoid(self.skip[dstntype]).unsqueeze(-1)
            if x_dst.shape != h.shape:
                h = h * alpha + (x_dst @ self.residual_w) * (1 - alpha)
            else:
                h = h * alpha + x_dst * (1 - alpha)
            if self.use_norm:
                h = self.norm(h)
            return h

    def message(self, edges):
        a, m = [], []
        etype = edges.data["etype"]
        k = torch.unbind(edges.src["k"], dim=1)
        q = torch.unbind(edges.dst["q"], dim=1)
        v = torch.unbind(edges.src["v"], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)  # (E, O)
            a.append(
                (kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d
            )  # (E,)
            m.append(self.relation_msg[i](v[i], etype, self.presorted))  # (E, O)
        return {"a": torch.stack(a, dim=1), "m": torch.stack(m, dim=1)}

    def reduce_sum(self, nodes):
        lam = 0.3
        _h = self.pre_reduction(
            nodes.data["h"], nodes.data["_TYPE"], sorted_by_type=True
        ).view(-1, self.num_heads, self.head_size)
        return {"h": (lam * torch.sum(nodes.mailbox["m"], dim=1)) + (1 - lam) * _h}
