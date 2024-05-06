import dgl
import pdb
import torch
from torch import nn
import torchtext
from collections import OrderedDict
from dgl.nn.pytorch.utils import JumpingKnowledge
from model_utils import Homogenizer, NodeEmbedding, EdgeEmbedding, TransposePool
from typing import List, Set, Dict, Tuple, Union, Optional, Callable
from utils import get_device
from redefined_modules import HGTConv

DEVICE = get_device()


class Classifier(nn.Module):
    def __init__(
        self,
        emb_input_dim: int,
        emb_output_dim: int,
        node_vocab_size: int,
        edge_vocab: Union[Set, List],
        padding_token_value: Optional[int],
        num_classes: int,
        pretrained: bool,
        freeze: bool,
        pretrained_embmat_dir: Optional[str],
        weight_edges: bool,
        add_self_loop: bool,
        allow_zero_in_degree: bool,
        **kwargs,
    ):
        super().__init__()
        self.allow_zero_in_degree = allow_zero_in_degree
        self.pretrained = pretrained
        self.homogenize = Homogenizer(edge_vocab)
        if not self.pretrained:
            if pretrained_embmat_dir is not None:
                mat = torch.load(pretrained_embmat_dir)
                self.node_embedding = NodeEmbedding(
                    emb_input_dim, node_vocab_size, freeze, mat
                )
            else:
                self.node_embedding = NodeEmbedding(
                    emb_input_dim,
                    node_vocab_size,
                    freeze,
                    padding_token_value=padding_token_value,
                )
        self.add_self_loop = add_self_loop
        self.weight_edges = weight_edges
        if self.weight_edges:
            self.edge_embedding = EdgeEmbedding(
                emb_input_dim, edge_vocab, purpose="dot"
            )

        self.tokens_pool = TransposePool(1, nn.AdaptiveMaxPool1d)
        self.nodes_pool = TransposePool(1, nn.AdaptiveAvgPool1d)
        self.linear = nn.Linear(emb_output_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        graph: dgl.DGLGraph,
        feat: torch.Tensor = None,
        eweight: torch.Tensor = None,
        get_attention: bool = False,
    ) -> torch.Tensor:
        g = graph
        if feat is not None:  # for use with GNNExplainer
            g.ndata["h"] = feat
        if "h" not in g.ndata.keys():  # h = node embeddings
            g = self.node_embedding(g)
        if self.weight_edges:
            if eweight is None:
                g = self.edge_embedding(
                    g
                )  # w = edge embeddings (not required for most models)
            else:
                g.edata["w"] = eweight
            if not g.is_homogeneous:
                g = self.homogenize(g)
        # homogenizing makes computation easier from here on out
        # we homogenize *after* getting embeddings, so that edges' types are represented in their embeddings
        if not g.is_homogeneous:
            g = self.homogenize(g, edata=[])
        if self.add_self_loop:
            g = dgl.add_self_loop(g)
        # commented out because our graphs *shouldn't* include nodes with zero in-degree
        # elif not self.allow_zero_in_degree:
        #     nonzero_nodes = torch.where(g.in_degrees()>0)[0]
        #     g = dgl.node_subgraph(g, nonzero_nodes, output_device=DEVICE)
        if g.ndata["h"].dim() > 2:
            g.ndata["h"] = self.tokens_pool(g.ndata["h"], dim0=1, dim1=2, squeeze=True)
        if (
            get_attention
        ):  # for post-hoc model examination purposes. can only be used with graph attention networks.
            _h, attention = self._forward(g, get_attention=True)
        else:
            _h = self._forward(g)
        # split graphs apart, to make one classification for each
        g.ndata["agged"] = _h
        gs = dgl.unbatch(g)
        _h = []
        for g in gs:
            # pool along number of nodes to get size (1, num_heads * head_size)
            pooled = self.nodes_pool(g.ndata["agged"], dim0=0, dim1=1, squeeze=True)
            _h.append(pooled)
        # shape is (num_graphs, emb_out_dim)
        _h = torch.cat(_h)
        if get_attention:
            return self.linear(_h).unsqueeze(0), attention
        return self.linear(_h).unsqueeze(0)


class GCNClassifier(Classifier):
    def __init__(
        self,
        emb_input_dim: int,
        hidden_dim: List[int],
        emb_output_dim: int,
        node_vocab_size: int,
        edge_vocab: Union[Set, List],
        num_classes: int = 2,
        pretrained: bool = False,
        freeze: bool = False,
        pretrained_embmat_dir: str = None,
        weight_edges: bool = False,
        add_self_loop: bool = False,
        dr: float = 0.0,
        padding_token_value: Optional[int] = None,
        conv_weight: bool = False,
        conv_bias: bool = False,
        allow_zero_in_degree: bool = False,
        linear_activation: bool = False,
        agg_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            node_vocab_size=node_vocab_size,
            edge_vocab=edge_vocab,
            pretrained=pretrained,
            freeze=freeze,
            pretrained_embmat_dir=pretrained_embmat_dir,
            weight_edges=weight_edges,
            num_classes=num_classes,
            add_self_loop=add_self_loop,
            padding_token_value=padding_token_value,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.convs, self.edge_pools = [], []
        dims = [emb_input_dim] + hidden_dim + [emb_output_dim]
        for i in range(len(dims) - 1):
            self.convs.append(
                dgl.nn.pytorch.GraphConv(
                    dims[i],
                    dims[i + 1],
                    weight=conv_weight,
                    bias=conv_bias,
                    allow_zero_in_degree=allow_zero_in_degree,
                ).to(DEVICE)
            )

        self.dropout = nn.Dropout(p=dr)
        self.agg_type = agg_type
        if self.agg_type is not None:
            self.jumping_knowledge = JumpingKnowledge(
                mode=self.agg_type,
                in_feats=emb_output_dim,
                num_layers=len(self.convs) + 1,
            )
        # linear_dims allows us to control the number and size of our linear layers
        if linear_dims is None:
            if self.agg_type == "cat":
                linear_dims = [sum(dims)]
            else:
                linear_dims = [emb_output_dim]
        linear_dims.append(num_classes)
        linear_dict = []
        for i in range(len(linear_dims) - 1):
            if linear_activation:
                linear_dict.append((f"activation_{i}", nn.Tanh()))
            linear_dict.append(
                (f"linear_{i}", nn.Linear(linear_dims[i], linear_dims[i + 1]))
            )
        self.linear = nn.Sequential(OrderedDict(linear_dict))

    def _forward(
        self,
        g: dgl.DGLGraph,
    ) -> torch.Tensor:
        _h = g.ndata["h"]
        _hs = [_h]
        if self.weight_edges:
            _w = g.edata["w"].squeeze(1)
        for i in range(len(self.convs)):
            if self.weight_edges:
                _h = self.convs[i](g, _h, edge_weight=_w)
            else:
                _h = self.convs[i](g, _h)
            _hs.append(_h)
        if self.agg_type is not None:
            return self.dropout(self.jumping_knowledge(_hs))
        return self.dropout(_hs[-1])


class HeteroGCNClassifier(Classifier):
    def __init__(
        self,
        emb_input_dim: int,
        hidden_dim: int,
        emb_output_dim: int,
        node_vocab_size: int,
        edge_vocab: Union[Set, List],
        ntypes: Union[Set, List],
        activation: Callable = nn.ReLU,
        num_classes: int = 2,
        dr: float = 0.0,
        num_convs: int = 5,
        pretrained: bool = False,
        freeze: bool = False,
        pretrained_embmat_dir: Optional[str] = None,
        weight_edges: bool = False,
        **kwargs,
    ):
        super().__init__(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            node_vocab_size=node_vocab_size,
            edge_vocab=edge_vocab,
            ntypes=ntypes,
            pretrained=pretrained,
            freeze=freeze,
            pretrained_embmat_dir=pretrained_embmat_dir,
            weight_edges=weight_edges,
        )
        if isinstance(edge_vocab, torchtext.vocab.Vocab):
            edge_vocab = edge_vocab.get_stoi()
        elif isinstance(edge_vocab, dict):
            edge_vocab = edge_vocab
        self.convs = []
        dims = [emb_input_dim] + hidden_dim + [emb_output_dim]
        for i in range(len(dims) - 1):
            self.convs.append(
                dgl.nn.pytorch.HeteroGraphConv(
                    {
                        etype: dgl.nn.pytorch.GraphConv(
                            dims[i], dims[i + 1], norm="both", allow_zero_in_degree=True
                        ).to(DEVICE)
                        for etype in edge_vocab
                    }
                )
            )
        self.activation = activation()
        # typed linear overrides super() linear
        self.linear = dgl.nn.pytorch.HeteroLinear(
            {ntype: emb_output_dim for ntype in ntypes}, num_classes
        )
        self.end_pool = TransposePool(1, nn.AdaptiveMaxPool1d)

    def forward(
        self,
        g: dgl.DGLGraph,
    ) -> torch.Tensor:
        if not self.pretrained:
            g = self.node_embedding(g)
        # g = self.edge_embedding(g)
        _h = g.ndata["h"]
        if isinstance(_h, torch.Tensor):  # only one ntype
            _h = {g.ntypes[0]: _h}

        for conv in self.convs:
            _h = conv(g, _h)

        _h = self.tokens_pool(_h, dim0=1, dim1=2)
        # shape should be (num_nodes, emb_dim, 1)
        _h = self.nodes_pool(_h, dim0=0, dim1=2)
        # shape should be (1, emb_dim, 1)
        # self.indices = indices
        for ntype in g.ntypes:
            _h[ntype] = torch.transpose(_h[ntype], dim0=0, dim1=2)
            _h[ntype] = torch.squeeze(_h[ntype])
        _h = self.linear(_h)
        _h = torch.stack([_h[key] for key in _h])[0]
        if _h.dim() > 1:
            _h = self.end_pool(_h, dim0=0, dim1=1)
        return _h.unsqueeze(0)


class HGTClassifier(Classifier):
    def __init__(
        self,
        emb_input_dim: int,
        emb_output_dim: int,
        node_vocab_size: int,
        edge_vocab: Union[Set, List],
        ntypes: Union[Set, List],
        num_classes: int = 2,
        dr: float = 0.0,
        pretrained: bool = False,
        pretrained_embmat_dir: str = None,
        freeze: bool = False,
        head_size: int = 100,
        num_heads: int = 4,
        num_convs: int = 3,
        weight_edges: bool = False,
        **kwargs,
    ):
        super().__init__(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            node_vocab_size=node_vocab_size,
            edge_vocab=edge_vocab,
            ntypes=ntypes,
            pretrained=pretrained,
            freeze=freeze,
            pretrained_embmat_dir=pretrained_embmat_dir,
            weight_edges=weight_edges,
        )
        ntype_to_i = {}
        for i, ntype in enumerate(ntypes):
            ntype_to_i[ntype] = i
        self.ntype_to_i = ntype_to_i

        if isinstance(edge_vocab, torchtext.vocab.Vocab):
            self.etype_to_i = edge_vocab.get_stoi()
        else:
            edge_to_i = {}
            for i, etype in enumerate(edge_vocab):
                edge_to_i[etype] = i
            self.etype_to_i = edge_to_i

        # self.conv = dgl.nn.pytorch.conv.HGTConv(emb_input_dim,
        self.conv = HGTConv(
            emb_input_dim,
            head_size,
            num_heads,
            len(ntypes),
            len(edge_vocab),
            dropout=dr,
        )
        self.num_convs = num_convs
        self.post_conv_pool = nn.AdaptiveMaxPool1d(head_size)

        self.linear = nn.Linear(head_size, num_classes)

    def _forward(
        self,
        g: dgl.DGLGraph,
    ) -> torch.Tensor:
        # get ntype and etype indices to appease HGTConv
        ntypes = g.ndata[dgl.NTYPE]
        etypes = g.edata["etype"]

        _h = g.ndata["h"]
        # put h through HGTConv
        for i in range(self.num_convs):
            _h = self.conv(g, _h, ntypes, etypes, presorted=True)
            _h = self.post_conv_pool(_h)
        # size is now (num_nodes, num_heads * head_size)

        return _h


class EGATClassifier(Classifier):
    def __init__(
        self,
        emb_input_dim: int,
        emb_output_dim: int,
        node_vocab_size: int,
        edge_vocab: Union[Set, List],
        num_heads: int,
        num_classes: int = 2,
        dr: float = 0.0,
        pretrained: bool = False,
        freeze: bool = False,
        pretrained_embmat_dir: str = None,
        add_self_loop: bool = False,
        padding_token_value: int = None,
        allow_zero_in_degree: bool = False,
        num_convs: int = 1,
        conv_bias: bool = False,
        agg_type: str = "cat",
        **kwargs,
    ):
        super().__init__(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            node_vocab_size=node_vocab_size,
            edge_vocab=edge_vocab,
            pretrained=pretrained,
            freeze=freeze,
            pretrained_embmat_dir=pretrained_embmat_dir,
            weight_edges=True,
            num_classes=num_classes,
            padding_token_value=padding_token_value,
            add_self_loop=add_self_loop,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.num_heads = num_heads
        self.num_convs = num_convs
        self.emb_input_dim = emb_input_dim
        in_dim = self.emb_input_dim
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(
                dgl.nn.pytorch.conv.EGATConv(
                    in_node_feats=in_dim,
                    in_edge_feats=in_dim,
                    out_node_feats=in_dim,
                    out_edge_feats=in_dim,
                    num_heads=num_heads,
                    # bias=conv_bias,
                )
            )

        self.agg_type = agg_type
        if self.agg_type is not None:
            self.jumping_knowledge = JumpingKnowledge(
                mode=self.agg_type,
                in_feats=emb_output_dim,
                num_layers=self.num_convs + 1,
            )
        self.dropout = torch.nn.Dropout(p=dr)
        # self.post_conv_norm = nn.LayerNorm(in_dim)
        if self.agg_type == "cat":
            self.linear = nn.Linear((self.num_convs + 1) * in_dim, num_classes)

    def _forward(
        self,
        g=dgl.DGLGraph,
        get_attention: bool = False,
    ) -> torch.Tensor:
        _h = g.ndata["h"]
        _hs = [_h]
        _w = g.edata["w"]
        for i in range(self.num_convs):
            _w = _w.squeeze()
            if get_attention:
                _h, _w, attention = self.convs[i](g, _h, _w, get_attention=True)
            else:
                _h, _w = self.convs[i](g, _h, _w, get_attention=False)
            # _h = self.post_conv_norm(_h.sum(dim=1))
            _h = _h.mean(dim=1)
            # _w = self.post_conv_norm(_w.sum(dim=1))
            _w = _w.mean(dim=1)
            _hs.append(_h)
        if get_attention:
            if self.agg_type is not None:
                return self.dropout(self.jumping_knowledge(_hs)), attention
            return _hs[-1], attention
        if self.agg_type is not None:
            return self.dropout(self.jumping_knowledge(_hs))
        return self.dropout(_hs[-1])


class GATClassifier(Classifier):
    def __init__(
        self,
        emb_input_dim: int,
        hidden_dim: int,
        emb_output_dim: int,
        node_vocab_size: int,
        edge_vocab: Union[Set, List],
        num_classes: int = 2,
        dr: float = 0.0,
        pretrained: bool = False,
        freeze: bool = False,
        pretrained_embmat_dir: str = None,
        weight_edges: bool = False,
        motifs_to_use: List[int] = [0, 1, 2],
        use_attention: bool = True,
        normalize_matrices: bool = True,
        num_heads: int = 1,
        add_self_loop: bool = False,
        padding_token_value: Optional[int] = None,
        allow_zero_in_degree: bool = False,
        **kwargs,
    ):
        super().__init__(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            node_vocab_size=node_vocab_size,
            edge_vocab=edge_vocab,
            pretrained=pretrained,
            freeze=freeze,
            pretrained_embmat_dir=pretrained_embmat_dir,
            weight_edges=weight_edges,
            num_classes=num_classes,
            padding_token_value=padding_token_value,
            add_self_loop=add_self_loop,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.motifs_to_use = motifs_to_use
        self.num_motifs = len(motifs_to_use)
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.normalize_matrices = normalize_matrices
        self.add_self_loop = add_self_loop
        self.hidden_dim = hidden_dim[0]

        self.W = nn.Linear(self.hidden_dim, self.hidden_dim, device=DEVICE)

        self.conv = dgl.nn.pytorch.GraphConv(emb_input_dim, self.hidden_dim)
        self.convs = []
        for k in range(self.num_motifs):
            self.convs.append(
                dgl.nn.pytorch.conv.DotGatConv(
                    in_feats=self.hidden_dim,
                    out_feats=self.hidden_dim,
                    num_heads=num_heads,
                    allow_zero_in_degree=True,
                ).to(DEVICE)
            )
        self.linear = nn.Linear(
            self.num_motifs * num_heads * hidden_dim[-1], num_classes, device=DEVICE
        )

    def reduced_edges(
        self,
        g: dgl.DGLGraph,
        new_adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        us, vs, eids = g.edges(form="all")
        out_eids = []
        node_pairs = list(zip(us, vs))
        for n, (i, j) in enumerate(node_pairs):
            if new_adj_mat[i][j] > 0:
                out_eids.append(eids[n])
        return out_eids

    def forward(
        self,
        g: dgl.DGLGraph,
        get_attention: bool = False,
    ) -> torch.Tensor:
        if "h" not in g.ndata.keys():
            g = self.node_embedding(g)
        if not g.is_homogeneous:
            g = self.homogenize(g, edata=[])
        # size is (num_nodes, num_tokens, emb_input_dim)
        _h = g.ndata["h"]
        if _h.dim() > 2:
            _h = self.tokens_pool(_h, dim0=1, dim1=2, squeeze=True)
        g.ndata["h"] = _h

        # get adjacency matrices
        if "0" not in g.ndata:
            g.ndata["0"] = g.adj()
        if not isinstance(g.ndata["0"], torch.Tensor):
            g.ndata["0"] = g.ndata["0"].to_dense()
        _A = g.ndata["0"]
        # message-passing
        _h = g.ndata["h"]
        Z = self.W(_A @ _h)
        g.ndata["h"] = Z
        _h_ks = []
        for i, k in enumerate(self.motifs_to_use):
            with g.local_scope():
                Z = g.ndata["h"]
                if k > 0:
                    A_tilde_k = g.ndata[str(k)]
                    if self.add_self_loop:
                        A_tilde_k = A_tilde_k + torch.eye(A_tilde_k.shape[0])
                    k_edges = self.reduced_edges(g, A_tilde_k)
                    if len(k_edges) == 0:  # no edges
                        _h_ks.append(
                            torch.zeros(
                                1,
                                self.convs[k]._num_heads * self.convs[k]._out_feats,
                                device=DEVICE,
                            )
                        )
                        continue
                    g.ndata["az"] = A_tilde_k @ Z
                    g_k = dgl.edge_subgraph(g, k_edges)
                else:
                    if self.add_self_loop:
                        g = dgl.add_self_loop(g)
                    g.ndata["az"] = Z
                    g_k = g
                if self.use_attention:
                    if get_attention:
                        g_k.ndata["h"], attention = self.convs[k](
                            g_k,
                            g_k.ndata["az"],
                            get_attention=True,
                        )
                        g_k.ndata["h"] = (
                            g_k.ndata["h"]
                            .view(-1, self.num_heads * self.hidden_dim)
                            .squeeze()
                        )
                    else:
                        g_k.ndata["h"] = (
                            self.convs[k](
                                g_k,
                                g_k.ndata["az"],
                            )
                            .view(-1, self.num_heads * self.hidden_dim)
                            .squeeze()
                        )
                g_ks = dgl.unbatch(g_k)
                _h = []
                for g_k in g_ks:
                    # pool along number of nodes to get size (1, num_heads * head_size)
                    pooled = self.nodes_pool(
                        g_k.ndata["h"], dim0=0, dim1=1, squeeze=True
                    )
                    _h.append(pooled)
                _h = torch.stack(_h)
                _h_ks.append(_h)
        _h = torch.cat(_h_ks, dim=1)
        if get_attention:
            return self.linear(_h), attention
        return self.linear(_h)


class MotifClassifier(Classifier):
    def __init__(
        self,
        emb_input_dim: int,
        hidden_dim: int,
        emb_output_dim: int,
        node_vocab_size: int,
        edge_vocab: Union[Set, List],
        ntypes: Union[Set, List],
        num_classes: int = 2,
        dr: float = 0.0,
        pretrained: bool = False,
        freeze: bool = False,
        pretrained_embmat_dir: str = None,
        weight_edges: bool = False,
        motifs_to_use: List[int] = [0, 1, 2],
        use_attention: bool = True,
        normalize_matrices: bool = True,
        num_heads: int = 1,
        add_self_loop: bool = False,
        **kwargs,
    ):
        super().__init__(
            emb_input_dim=emb_input_dim,
            emb_output_dim=emb_output_dim,
            node_vocab_size=node_vocab_size,
            edge_vocab=edge_vocab,
            ntypes=ntypes,
            pretrained=pretrained,
            freeze=freeze,
            pretrained_embmat_dir=pretrained_embmat_dir,
            weight_edges=weight_edges,
            num_classes=num_classes,
        )
        self.hidden_dim = hidden_dim[0]
        self.num_heads = num_heads if use_attention else 1
        self.num_motifs = len(motifs_to_use)
        self.num_classes = num_classes
        self.motifs_to_use = motifs_to_use
        self.use_attention = use_attention
        self.normalize_matrices = normalize_matrices
        self.add_self_loop = add_self_loop

        self.conv = dgl.nn.pytorch.GraphConv(emb_input_dim, emb_input_dim)
        self.W = nn.Linear(emb_input_dim, self.hidden_dim, device=DEVICE)
        if use_attention:
            self.attn_func = nn.Linear(self.hidden_dim * 2, 1, device=DEVICE)
        self.W_k = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim)
                for k in range(self.num_motifs)
            ]
        )  # indiv weight matrices for each motif
        self.tanh = torch.nn.Tanh()
        self.linear = nn.Linear(
            (self.num_motifs + 1) * hidden_dim[-1] * self.num_heads,
            num_classes,
            device=DEVICE,
        )
        # self.linear = nn.Linear(hidden_dim[-1] * self.num_heads, num_classes, device=DEVICE)
        # self.dropout = dgl.transforms.DropEdge(p=dr)
        self.dropout = torch.nn.Dropout(p=dr)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        # adapted from https://docs.dgl.ai/en/1.1.x/tutorials/models/1_gnn/9_gat.html
        _a = self.attn_func(torch.cat([edges.src["h"], edges.dst["h"]], dim=1))
        to_return = torch.nn.functional.leaky_relu(_a).squeeze()
        return {"e": to_return}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        # adapted from https://docs.dgl.ai/en/1.1.x/tutorials/models/1_gnn/9_gat.html
        edges._eid = torch.tensor(
            [i for i in range(edges.batch_size())], device=DEVICE
        ).to(int)
        us, vs, _ = edges.edges()
        # edge_attn = torch.zeros_like(edges.src["az"], device=DEVICE)
        edge_attn = torch.zeros(edges.batch_size())
        # TODO like edata_to_attnmat, could be made more efficient with sparse mat
        for n, (u, v) in enumerate(zip(us, vs)):
            edge_attn[n] = edges.src["attn"][n][v]  # this will be broadcasted
        # TODO use edge weights, * by edges.src['az']?
        return {"m": edges.dst["az"], "e": edge_attn}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # adapted from https://docs.dgl.ai/en/1.1.x/tutorials/models/1_gnn/9_gat.html
        # attn = torch.nn.functional.softmax(nodes.mailbox["e"], dim=0)
        attn = nodes.mailbox["e"].to(DEVICE).unsqueeze(2)
        _h = torch.mean(attn * nodes.mailbox["m"], dim=1)  # cat instead?
        return {"h": _h}

    def edata_to_attnmat(self, g: dgl.DGLGraph) -> torch.Tensor:
        edata = g.edata["e"]
        us, vs = g.edges(form="uv")
        indices = torch.stack([us, vs])
        # TODO this might be more efficient in sparse format
        out = torch.zeros(g.num_nodes(), g.num_nodes(), device=DEVICE)
        for n in range(len(edata)):
            out[us[n]][vs[n]] = edata[n]
        return out

    def reduced_edges(
        self,
        g: dgl.DGLGraph,
        new_adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        us, vs, eids = g.edges(form="all")
        out_eids = []
        node_pairs = list(zip(us, vs))
        for n, (i, j) in enumerate(node_pairs):
            if new_adj_mat[i][j] > 0:
                out_eids.append(eids[n])
        return out_eids

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        if "h" not in g.ndata.keys():
            g = self.node_embedding(g)
        if self.weight_edges:
            g = self.edge_embedding(g)
            if not g.is_homogeneous:
                # g = dgl.to_homogeneous(g, ndata="h", edata="w", store_type=True)
                g = self.homogenize(g)
            # _w = g.edata["w"]
        elif not g.is_homogeneous:
            # g = dgl.to_homogeneous(g, ndata="h", store_type=True)
            g = self.homogenize(g, edata=[])
        # size is (num_nodes, num_tokens, emb_input_dim)
        _h = g.ndata["h"]
        if _h.dim() > 2:
            _h = self.tokens_pool(_h, dim0=1, dim1=2, squeeze=True)
        g.ndata["h"] = _h

        # get adjacency matrices
        if "0" not in g.ndata:
            g.ndata["0"] = g.adj().to_dense()
        _A = g.ndata["0"]
        # message-passing
        _h = g.ndata["h"]
        _h = self.conv(dgl.add_self_loop(g), _h)
        Z = self.W(_h)  # Z shape = (n_nodes, hidden_dim)
        g.ndata["h"] = Z
        _h_ks = [self.nodes_pool(Z, dim0=0, dim1=1, squeeze=False).t()]
        for i, k in enumerate(self.motifs_to_use):
            Z = g.ndata["h"]
            A_k = g.ndata[str(k)]
            k_edges = self.reduced_edges(g, A_k)
            if len(k_edges) == 0:  # no edges
                _h_ks.append(
                    torch.zeros(
                        len(dgl.unbatch(g)), self.W.weight.shape[1], device=DEVICE
                    )
                )
                # _h_ks.append(torch.zeros(len(dgl.unbatch(g)), self.num_classes, device=DEVICE))
                continue
            g.ndata["az"] = self.dropout(self.W_k[i](Z))
            # AZ shape = (hidden_dim, num_nodes)
            # batched graphs must get subgraphed separately in order to keep batched format
            gs = dgl.unbatch(g)
            if len(gs) > 1:
                to_batch = []
                for _g in gs:
                    k_edges = self.reduced_edges(_g, A_k)
                    sub_g = dgl.edge_subgraph(g, k_edges)
                    if self.add_self_loop:
                        sub_g = dgl.add_self_loop(sub_g)
                    to_batch.append(sub_g)
                g_k = dgl.batch(to_batch)
                if sum(g_k.batch_num_edges()) == 0:
                    _h_ks.append(
                        torch.zeros(len(gs), self.W.weight.shape[1], device=DEVICE)
                    )
                    continue
            else:
                k_edges = self.reduced_edges(g, A_k)
                g_k = dgl.edge_subgraph(g, k_edges)
                if self.add_self_loop:
                    g_k = dgl.add_self_loop(g_k)
            if self.use_attention:
                g_k.apply_edges(self.edge_attention)
                g_k.ndata["attn"] = self.edata_to_attnmat(g_k)  # .to_dense()
            else:
                g_k.ndata["attn"] = g_k.adj().to_dense()  # A_k
            g_k.ndata["attn"] = self.tanh(g_k.ndata["attn"])
            g_k.update_all(
                self.message_func, self.reduce_func
            )  # motif-wise aggregation
            g_ks = dgl.unbatch(g_k)
            _h = []
            for g_k in g_ks:
                pooled = self.nodes_pool(g_k.ndata["h"], dim0=0, dim1=1, squeeze=True)
                # shape is (emb_dim)
                _h.append(pooled)
            _h = torch.stack(_h)
            # shape is (batch_size, emb_dim)
            _h_ks.append(_h)
        _h = torch.cat(_h_ks, dim=1)
        # shape should be (num_motifs, batch_size, num_classes)
        # TODO redundancy minimization
        # TODO do this on node level?
        # non_redundant_h_ks = []
        # for i, k in enumerate(self.motifs_to_use):
        #     H_bar_k = torch.cat([_h[:i], _h[i+1:], Z]).view(-1, 13 * self.hidden_dim)
        # diverge from original paper here --- they keep nodes discrete, bc they're doing node classification
        # return self.nodes_pool(_h, dim0=0, dim1=1, squeeze=False).t()
        # return self.nodes_pool(_h, dim0=1, dim1=2, squeeze=False).squeeze(-1)
        return self.linear(_h)
