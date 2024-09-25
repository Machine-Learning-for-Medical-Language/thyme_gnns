import dgl
import pdb
import torch
import torchtext

from torch import nn
from typing import Callable, Dict, Optional, Tuple, Union

from utils import get_device

DEVICE = get_device()


# for overriding dgl's homogenization.
# need to make this a full class so we can initialize it with edge vocab.
# TODO node types as well? not really an issue
class Homogenizer(nn.Module):
    def __init__(
        self,
        edge_vocab: torchtext.vocab.Vocab,
    ):
        super().__init__()
        if isinstance(edge_vocab, dict):
            self.stoi = edge_vocab
        elif isinstance(edge_vocab, torchtext.vocab.Vocab):
            self.stoi = edge_vocab.get_stoi()
        else:
            self.stoi = {etype: i for i, etype in enumerate(edge_vocab)}

    def forward(
        self,
        g: dgl.DGLGraph,
        ndata: Optional[list] = None,
        edata: Optional[list] = None,
        store_type: bool = True,
    ) -> dgl.DGLGraph:
        if store_type:
            itypes = {}
            for etype in g.canonical_etypes:
                itypes[etype] = (
                    torch.tensor([self.stoi[etype]])
                    .repeat(len(g[etype].edges()[0]), 1)
                    .to(DEVICE)
                )
            g.edata["etype"] = itypes
        # by default, keep all ndata and edata
        if ndata is None:
            ndata = g.ndata.keys()
        if edata is None:
            edata = g.edata.keys()  # this now includes etype
        return dgl.to_homogeneous(g, ndata=ndata, edata=edata)


class NodeEmbedding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        vocab_len: int,
        freeze: bool,
        mat: torch.Tensor = None,
        padding_token_value: Optional[int] = None,
    ):
        super().__init__()
        if mat is not None:
            self.embed = nn.Embedding.from_pretrained(mat, freeze=freeze).to(DEVICE)
        elif padding_token_value is not None:
            self.embed = nn.Embedding(
                vocab_len, emb_size, _freeze=freeze, padding_idx=0
            ).to(DEVICE)
            with torch.no_grad():
                self.embed.weight[0] = (
                    padding_token_value  # empty token gets xt low value to minimize influence
                )
        else:
            self.embed = nn.Embedding(vocab_len, emb_size, _freeze=freeze).to(DEVICE)

    def forward(
        self,
        g: dgl.DGLGraph,
    ) -> dgl.DGLGraph:
        if len(g.ntypes) == 1:
            g.ndata["h"] = self.embed(g.ndata["tokens"])
        else:
            for ntype in g.ndata["tokens"].keys():
                g.nodes[ntype].data["h"] = self.embed(g.ndata["tokens"][ntype])
        return g


class EdgeEmbedding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        edge_vocab: torchtext.vocab.Vocab,
        homogeneous: bool = False,
        purpose: str = "matmul",
    ):
        super().__init__()
        embeds = []
        if homogeneous or isinstance(edge_vocab, dict):
            self.stoi = edge_vocab
        elif isinstance(edge_vocab, torchtext.vocab.Vocab):
            self.stoi = edge_vocab.get_stoi()
        else:
            self.stoi = {etype: i for i, etype in enumerate(edge_vocab)}
        if purpose == "matmul":
            x_emb_size = emb_size
            for etype in self.stoi:
                embeds.append(nn.Embedding(x_emb_size, emb_size).weight.to(DEVICE))
            self.embed = nn.Parameter(torch.stack(embeds))
        elif purpose == "dot":
            self.embed = nn.Embedding(len(self.stoi), emb_size).to(DEVICE)

    def forward(
        self,
        g: dgl.DGLGraph,
    ) -> dgl.DGLGraph:
        if g.is_homogeneous:
            etype_embeddings = []
            for etype in g.edata[dgl.ETYPE]:
                etype_embeddings.append(self.embed.weight[etype])
            etype_embeddings = torch.stack(etype_embeddings)
            g.edata["w"] = etype_embeddings
            return g
        embed_dct = {}
        for etype in g.canonical_etypes:
            try:
                i = self.stoi[etype]
            except KeyError:
                i = self.stoi[(etype[0], "<unk>", etype[2])]
            if isinstance(self.embed, nn.Parameter):
                etype_embed = (
                    self.embed[i].unsqueeze(0).repeat(g[etype].num_edges(), 1, 1)
                )
            elif isinstance(self.embed, nn.Embedding):
                etype_embed = (
                    self.embed.weight[i].unsqueeze(0).repeat(g[etype].num_edges(), 1, 1)
                )
            embed_dct[etype] = etype_embed
        g.edata["w"] = embed_dct
        return g


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        emb_size: int = 768,
        freeze: bool = False,
        f: Callable = torch.sin,
    ):
        super().__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(1), requires_grad=not freeze)
        self.b0 = nn.parameter.Parameter(torch.randn(1), requires_grad=not freeze)
        self.w = nn.parameter.Parameter(
            torch.randn(emb_size - 1), requires_grad=not freeze
        )
        self.b = nn.parameter.Parameter(
            torch.randn(emb_size - 1), requires_grad=not freeze
        )
        self.f = f

    def time2vec(self, input: torch.Tensor):
        input = input.unsqueeze(1)
        out = [
            ((input * self.w0) + self.b0),
            self.f(torch.matmul(input, self.w.unsqueeze(0)) + self.b),
        ]
        return torch.cat(out, 1)

    def forward(self, g: dgl.DGLGraph):
        g.nodes["normed_time"].data["h"] = self.time2vec(
            g.ndata["relative_time"]["normed_time"]
        )
        return g


class TransposePool(nn.Module):
    def __init__(
        self,
        pool_out_size: int = 1,
        pool_module: Optional[Callable] = nn.AdaptiveAvgPool1d,
        return_indices: bool = False,
    ):
        super().__init__()
        self.pool_out_size = pool_out_size
        self.return_indices = return_indices
        if pool_module == nn.AdaptiveMaxPool1d:
            self.pool = pool_module(pool_out_size, self.return_indices)
        else:
            self.pool = pool_module(pool_out_size)

    def forward(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        dim0: int = 0,
        dim1: int = 2,
        squeeze: bool = False,
    ) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            input = torch.transpose(input, dim0, dim1)
            if self.return_indices:
                pooled, indices = self.pool(input)
            else:
                pooled = self.pool(input)
            if squeeze:
                pooled = torch.squeeze(pooled)
            if self.return_indices:
                return pooled, indices
            else:
                return pooled
        elif isinstance(input, dict):
            if self.return_indices:
                indices = {}
            for key in input:
                transposed = torch.transpose(input[key], dim0, dim1)
                if self.return_indices:
                    input[key], indices[key] = self.pool(transposed)
                else:
                    input[key] = self.pool(transposed)
                if squeeze:
                    input[key] = torch.squeeze(input[key])
            return input


class EdgePool(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
    ):
        super().__init__()
        if in_size != out_size:
            self.pool = nn.Linear(in_size, out_size)
        else:
            self.pool = lambda x: x

    def forward(
        self,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        return self.pool(feats)
