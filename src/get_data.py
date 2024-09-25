from collections import defaultdict
import os

os.environ["DGLBACKEND"] = "pytorch"
import copy
import dgl
import json
import logging
import math
import numpy as np
import pandas as pd
import pdb
import pickle
import scipy
import torch
import torchtext

from dgl.dataloading import GraphDataLoader
from model_utils import Homogenizer
from scipy.linalg import block_diag
from temporal_closure import extrapolate
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from typing import Any, List, Set, Dict, Tuple, Union, Optional, Callable
from utils import get_device


DEVICE = get_device()
# DEVICE = "cpu"

doctime_etypes = {
    ("event", "AFTER", "DocTime"),
    ("event", "BEFORE", "DocTime"),
    ("event", "BEFORE/OVERLAP", "DocTime"),
    ("event", "OVERLAP", "DocTime"),
}


def load_graph(
    file_path: str,
    loader: callable,  # loader should be e.g. `json.load`, `pickle.load`
) -> Dict:
    try:
        with open(file_path, "rb") as f:
            data = loader(f)
    except:
        return None
    if "relations" not in data or len(data["relations"]) == 0:
        return None
    data["timex"] = data["timexes"]
    data["event"] = data["events"]
    del data["events"], data["timexes"]
    return data


def load_csv(
    csv_path: str,
    columns: List[str],
    data_path: str,
    num_train_data: Optional[int],
    num_dev_data: Optional[int],
    num_test_data: Optional[int],
    oversample: Optional[float | List[float]] = None,
) -> Union[List[int], List[str]]:
    columns = columns + ["note_id"] if "note_id" not in columns else columns
    # match note names in df to files in directory
    df = pd.read_csv(csv_path)
    files = os.listdir(data_path)
    # normalize file names to match note names in df
    if "_" in files[0]:  # in case files have a prefix, e.g. `mimic_123456.json`
        files = [f.split("_")[1].split(".")[0] for f in os.listdir(data_path)]
    else:
        files = [f.split(".")[0] for f in os.listdir(data_path)]
    df = df.loc[
        df["note_id"].isin(files)
    ]  # only select notes that we've already processed
    # do oversampling last so we don't affect the label distribution in the dev/test sets
    if oversample is not None:
        if isinstance(oversample, list):
            if len(oversample) == 1:  # if only minority is provided
                oversample = [1 - oversample[0]] + oversample
        elif isinstance(oversample, float):
            oversample = [1 - oversample, oversample]
        train_dfs = []
        dfs = []
        for i, group in df.groupby(columns[0]):
            n = round(oversample[i] * num_train_data)
            train_dfs.append(group[:n])
            dfs.append(group[n:])
        train_df = pd.concat(train_dfs)[columns]
        train_df = train_df.sample(frac=1, random_state=20)
        df = pd.concat(dfs)
        df = df.sample(frac=1, random_state=20)
    else:
        train_df = df[:num_train_data]
        df = df[num_train_data:]
    dev_df = df[:num_dev_data][columns]
    df = df[num_dev_data:]
    test_df = df[:num_test_data][columns]
    return train_df, dev_df, test_df


def load_data_dir(
    data_dir: str,
    label_dir: str,
    pretrained_dir: Optional[str],
    labels: List[str],
    oversample: Optional[List[float]] = None,
    num_train_data: int = 0,
    num_dev_data: int = 0,
    num_test_data: int = 0,
    format: str = "pickle",
) -> Tuple[List[int], List[Dict]]:
    data_path = pretrained_dir if pretrained_dir is not None else data_dir
    train_labels, dev_labels, test_labels = load_csv(
        csv_path=label_dir,
        columns=labels + ["note_id"],
        data_path=data_path,
        oversample=oversample,
        num_train_data=num_train_data,
        num_dev_data=num_dev_data,
        num_test_data=num_test_data,
    )
    out = []
    if format in ["pickle", "pkl"]:
        loader = pickle.load
    elif format == "json":
        loader = json.load
    for df in [train_labels, dev_labels, test_labels]:
        fnames = df["note_id"].values
        logging.info("Loading data")
        graphs = []
        for fname in tqdm(fnames):
            if os.path.isfile(os.path.join(data_dir, f"{fname}.pkl")):
                graph = load_graph(os.path.join(data_dir, f"{fname}.pkl"), loader)
                if graph is None:
                    df = df.loc[~df["note_id"].isin([fname])]
                    continue
                graphs.append(graph)
        out.append((df["note_id"].values, df[labels].values, graphs))
    return out


def get_vocab_and_pipeline(
    train_data: Dict[str, Dict], temporal_closure: bool
) -> Tuple[torchtext.vocab.Vocab, Callable]:
    tokenizer = get_tokenizer("basic_english")

    def tokenize(strlist):
        return list(map(tokenizer, strlist))

    def yield_tokens(
        data_iter,
    ):  # given a graph, get all of its entities, and tokenize the text of all of those entities
        for dct in data_iter:
            all_entities = {}
            all_entities.update(dct["event"])
            if "timex" in dct:
                all_entities.update(dct["timex"])
            for entity in all_entities:
                yield tokenizer(all_entities[entity]["text"])

    vocab = torchtext.vocab.build_vocab_from_iterator(
        yield_tokens(iter(copy.deepcopy(train_data))), specials=["<pad>", "<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    # note all observed entity-entity relations, including between all events and DocTime
    def get_edge_categories(data_iter):
        # hardcode unknown relations
        edge_categories = {
            ("event", "<unk>", "event"),
            ("event", "<unk>", "timex"),
            ("timex", "<unk>", "event"),
        }
        if temporal_closure:
            edge_categories.add(("timex", "<unk>", "timex"))
        for dct in data_iter:
            for event in dct["event"]:
                if event.lower().startswith(
                    "timex"
                ):  # time expressions don't have doctimerel
                    continue
                edge_categories.add(("event", dct["event"][event]["dtr"], "DocTime"))
                edge_categories.add(
                    ("DocTime", category_reverse(dct["event"][event]["dtr"]), "event")
                )
                if temporal_closure:
                    edge_categories.add(
                        ("timex", dct["event"][event]["dtr"], "DocTime")
                    )
                    edge_categories.add(
                        (
                            "DocTime",
                            category_reverse(dct["event"][event]["dtr"]),
                            "timex",
                        )
                    )
            if "normed_time" in dct:
                edge_categories.update(
                    {
                        ("event", "<unk>", "normed_timex"),
                        ("normed_timex", "<unk>", "normed_timex"),
                    }
                )
                for normed_time in dct["normed_time"]:
                    edge_categories.add(
                        (
                            "normed_time",
                            dct["normed_time"][normed_time]["dtr"],
                            "DocTime",
                        )
                    )
                    edge_categories.add(
                        (
                            "DocTime",
                            category_reverse(dct["normed_time"][normed_time]["dtr"]),
                            "normed_time",
                        )
                    )
            for relation in dct["relations"]:  # inter-entity relations
                arg1_type = "_".join(relation["arg1"].split("_")[:-1]).lower()
                arg2_type = "_".join(relation["arg2"].split("_")[:-1]).lower()
                category = relation["category"]
                reverse_category = category_reverse(category)
                edge_categories.add((arg1_type, category, arg2_type))
                edge_categories.add((arg2_type, reverse_category, arg1_type))
                if temporal_closure:
                    edge_categories.add(("timex", category, "timex"))
                    edge_categories.add(("timex", reverse_category, "timex"))

        edge_categories = {
            cat: i for i, cat in enumerate(edge_categories)
        }  # edge type to i
        return edge_categories

    edge_vocab = get_edge_categories(copy.deepcopy(train_data))

    def text_pipeline(x, max_len):
        tokens = tokenize(x)
        vocab_lists = []
        for token_list in tokens:
            token_list = token_list[:max_len]
            while len(token_list) < max_len:
                token_list.append("<pad>")
            vocab_lists.append(vocab(token_list))
        return torch.LongTensor(vocab_lists)

    return vocab, edge_vocab, text_pipeline


def category_reverse(category: str) -> str:
    if category == "OVERLAP":
        return category
    elif category == "BEFORE":
        return "AFTER"
    elif category == "AFTER":
        return "BEFORE"
    elif category.endswith("-1"):
        return category[:-2]
    else:
        return category + "-1"


def make_relations(
    datum: Dict,
    known_etypes: List,
    pretrained: bool = False,
) -> Dict:
    entity_ids = defaultdict(lambda: {})
    entity_ids["DocTime"] = {"DocTime": 0}
    graph_data = defaultdict(lambda: {"sources": [], "destinations": []})
    datum["event"] = {
        event: datum["event"][event]
        for event in datum["event"]
        if (not event.startswith("Timex"))
        and (
            datum["event"][event]["text"].isascii() or pretrained
        )  # if bert can handle a string, we don't care if it's ascii or not
    }
    for i, event in enumerate(datum["event"]):
        entity_ids["event"][event] = i
        # get doctimerel
        doctimerel = datum["event"][event]["dtr"]
        graph_data[("event", doctimerel, "DocTime")]["sources"].append(i)
        graph_data[("event", doctimerel, "DocTime")]["destinations"].append(
            0
        )  # idx of DocTime node
        reverse_doctimerel = category_reverse(doctimerel)
        graph_data[("DocTime", reverse_doctimerel, "event")]["sources"].append(
            0
        )  # idx of DocTime node
        graph_data[("DocTime", reverse_doctimerel, "event")]["destinations"].append(i)
    if "normed_time" in datum:
        for i, normed_time in enumerate(datum["normed_time"]):
            entity_ids["normed_time"][normed_time] = i
            # get doctimerel
            doctimerel = datum["normed_time"][normed_time]["dtr"]
            graph_data[("normed_time", doctimerel, "DocTime")]["sources"].append(i)
            graph_data[("normed_time", doctimerel, "DocTime")]["destinations"].append(
                0
            )  # idx of DocTime node
            reverse_doctimerel = category_reverse(doctimerel)
            graph_data[("DocTime", reverse_doctimerel, "normed_time")][
                "sources"
            ].append(
                0
            )  # idx of DocTime node
            graph_data[("DocTime", reverse_doctimerel, "normed_time")][
                "destinations"
            ].append(i)
    for relation in datum["relations"]:
        arg1 = relation["arg1"]
        arg2 = relation["arg2"]
        arg1_type = "_".join(arg1.split("_")[:-1]).lower()
        arg2_type = "_".join(arg2.split("_")[:-1]).lower()
        if (
            (arg1_type == "event" and arg1 not in entity_ids["event"])
            or (arg2_type == "event" and arg2 not in entity_ids["event"])
            or (arg1_type == "normed_time" and arg1 not in entity_ids["normed_time"])
            or (arg2_type == "normed_time" and arg2 not in entity_ids["normed_time"])
        ):
            continue

        # timexes get added here to eliminate those that are not actually used
        # (events are all used, bc of their DocTimeRel)
        if arg1_type == "timex":
            if arg2_type == "timex":
                continue
            if arg1 not in entity_ids["timex"]:
                entity_ids["timex"][arg1] = len(entity_ids["timex"])
        elif arg2_type == "timex":
            if arg2 not in entity_ids["timex"]:
                entity_ids["timex"][arg2] = len(entity_ids["timex"])
        arg1_id = entity_ids[arg1_type][relation["arg1"]]
        arg2_id = entity_ids[arg2_type][relation["arg2"]]
        if (arg1_type, relation["category"], arg2_type) in known_etypes:
            category = relation["category"]
            reverse_category = category_reverse(category)
        else:  # if etype has not been seen in training, replace with default token
            category = "<unk>"
            reverse_category = "<unk>"
        graph_data[(arg1_type, category, arg2_type)]["sources"].append(arg1_id)
        graph_data[(arg1_type, category, arg2_type)]["destinations"].append(arg2_id)
        graph_data[(arg2_type, reverse_category, arg1_type)]["sources"].append(
            arg2_id
        )  # add reverse so we can update source nodes in add'n to dest nodes
        graph_data[(arg2_type, reverse_category, arg1_type)]["destinations"].append(
            arg1_id
        )
    return graph_data, entity_ids


# turn dict into graphs
def graphify(
    data: List[Dict],
    edge_vocab: torchtext.vocab.Vocab,
    text_pipeline: Callable,
    max_len: int = 4,
    pretrained: bool = False,
    temporal_closure: bool = False,
) -> List[dgl.DGLGraph]:
    logging.info("Graphifying data")
    out = []
    homogenizer = Homogenizer(edge_vocab)
    for datum in tqdm(data):  # datum = one graph/one file
        if "timex" in datum:
            datum["timex"] = {
                timex: datum["timex"][timex]
                for timex in datum["timex"]
                if timex.lower().startswith("timex")
            }
        datum["event"] = {
            event: datum["event"][event]
            for event in datum["event"]
            if event.lower().startswith("event")
        }
        if temporal_closure:
            graph_data, entity_ids = extrapolate(datum)
            graph_data = {
                etype: (graph_data[etype][0].to(int), graph_data[etype][1].to(int))
                for etype in graph_data
            }  # to appease dgl
        else:
            graph_data, entity_ids = make_relations(
                datum, edge_vocab, pretrained=pretrained
            )
            # just node indices, representing edges between them
            for etype in graph_data:
                graph_data[etype] = (
                    torch.tensor(graph_data[etype]["sources"]),
                    torch.tensor(graph_data[etype]["destinations"]),
                )
        # get number of nodes in each category
        num_nodes = 0
        for ntype in entity_ids:
            max_id = 0
            for etype in graph_data:
                # check for maximum id
                # only need to check first entity in relation bc graph is bidirectional;
                # all nodes that are destinations are also sources
                if etype[0] == ntype:
                    if torch.max(graph_data[etype][0]).item() > max_id:
                        max_id = torch.max(graph_data[etype][0]).item()
                if etype[2] == ntype:
                    if torch.max(graph_data[etype][1]).item() > max_id:
                        max_id = torch.max(graph_data[etype][1]).item()
            if (
                max_id > 0
            ):  # if this ntype actually exists in this data, add 1 to avoid off-by-1 errors
                num_nodes += max_id + 1
        if num_nodes == 0:  # empty graph
            continue
        g = dgl.heterograph(graph_data).to(DEVICE)
        if (homogenizer(g, ndata=[], edata=[]).in_degrees() == 0).any():
            pdb.set_trace()
        if not pretrained:
            for ntype in g.ntypes:
                if (
                    ntype in datum
                    and isinstance(datum[ntype], dict)
                    and ntype != "normed_time"
                ):
                    # eliminate nodes that are isolated (cannot be used by algorithm)
                    old_datum_ntype = datum[ntype]
                    datum[ntype] = {
                        ent: datum[ntype][ent]
                        for ent in datum[ntype]
                        if ent in entity_ids[ntype]
                        and entity_ids[ntype][ent] in g.nodes(ntype)
                    }
                    if len(old_datum_ntype) > len(datum[ntype]):  # and ntype!="timex":
                        pdb.set_trace()
                    # get text for each node
                    # put text through pipeline to get tokens
                    g.nodes[ntype].data["tokens"] = text_pipeline(
                        [datum[ntype][ent]["text"] for ent in datum[ntype]], max_len
                    ).to(DEVICE)

                elif ntype == "DocTime":
                    g.nodes["DocTime"].data["tokens"] = text_pipeline(
                        ["<pad>"], max_len
                    ).to(DEVICE)
        if "normed_time" in g.ntypes:
            g.nodes["normed_time"].data["relative_time"] = torch.tensor(
                [
                    datum["normed_time"][ent]["relative_time"]
                    for ent in datum["normed_time"]
                ],
                device=DEVICE,
            )
        # if (g["timex"].in_degrees() == 0).any():
        #     isolated_nodes = g.in_degrees()==0
        #     g.remove_node(isolated_nodes, ntype="timex")
        #     pdb.set_trace()
        out.append(g)
    return out


def get_loader(
    data: List[Tuple[str, Dict, Any]],
    label_pipeline: Callable,
    batch_size: int,
    homogenize: bool,
    edge_vocab,
    shuffle: bool = False,
    pretrained_dir: Optional[str] = None,
    emb_dim: int = 100,
    motifs_to_use: List[int] = [0],
    motifs_dir: Optional[str] = None,
) -> GraphDataLoader:
    def collate_batch(batch):
        fnames, labels, graphs = zip(*batch)
        if homogenize:
            homogenizer = Homogenizer(edge_vocab)
        if pretrained_dir is not None:
            # load embeddings
            for i, g in enumerate(graphs):
                # TODO merge with load_graph above
                with open(os.path.join(pretrained_dir, f"{fnames[i]}.pkl"), "rb") as f:
                    embeddings = pickle.load(f)
                embeddings = {
                    ntype: embeddings[ntype]
                    for ntype in embeddings
                    if ntype in g.ntypes and len(embeddings[ntype]) > 0
                }
                for ntype in embeddings:
                    embeddings[ntype] = embeddings[ntype].to(DEVICE)
                # if "DocTime" not in embeddings:
                #     embeddings["DocTime"] = (
                #         torch.ones((1, 1, emb_dim)).to(DEVICE).detach()
                #     )
                try:
                    for ntype in g.ntypes:
                        # if "tokens" in g.nodes[ntype].data.keys():
                        if ntype in embeddings:
                            # g.ndata["h"][ntype] = embeddings[ntype]
                            g.nodes[ntype].data["h"] = embeddings[ntype]
                except:
                    pdb.set_trace()
                if homogenize:
                    g = homogenizer(g)
        elif homogenize:
            graphs = [homogenizer(g, ndata=["tokens"]) for g in graphs]
        motif_adjs_graph_level = {motif: [] for motif in motifs_to_use}
        # read motif adj matrices
        if motifs_dir is not None and len(motifs_to_use) > 0 and motifs_to_use != [0]:
            for i, graph in enumerate(graphs):
                for motif in motifs_to_use:
                    if motif == 0:
                        motif_adj = graph.adj().to_dense()
                    else:
                        motif_adj_info = torch.load(
                            os.path.join(motifs_dir, str(motif), fnames[i] + ".pt")
                        )
                        shape = graph.adj().to_dense().shape
                        if len(motif_adj_info["row"]) > 0:
                            indices = (
                                motif_adj_info["row"].to(int).numpy(),
                                motif_adj_info["col"].to(int).numpy(),
                            )
                            motif_adj = torch.tensor(
                                scipy.sparse.coo_array(
                                    (motif_adj_info["val"].squeeze(), indices),
                                    shape=shape,
                                ).todense(),
                                device=DEVICE,
                            )
                        else:
                            motif_adj = torch.zeros_like(graph.adj().to_dense())
                        if torch.any(motif_adj != motif_adj.t()):
                            motif_adj = (
                                motif_adj + motif_adj.t()
                            )  # make adj mat undirected
                    motif_adjs_graph_level[motif].append(motif_adj.cpu())
            batched_graph = dgl.batch(graphs)
            for motif in motifs_to_use:
                batched_graph.ndata[str(motif)] = torch.tensor(
                    block_diag(*motif_adjs_graph_level[motif]), device=DEVICE
                )
        elif len(graphs) > 1:
            batched_graph = dgl.batch(graphs)
        else:
            batched_graph = graphs[0]
        labels = torch.tensor(
            np.array(label_pipeline(labels))
        )  # np.array(labels) bc making a tensor from a list of arrays is very slow
        return fnames, labels, batched_graph

    return dgl.dataloading.GraphDataLoader(
        data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
    )


def get_data_loaders(
    data_dir: str,
    labels_path: str,
    pretrained_dir: Optional[str],
    task_names: List[str],
    batch_size: int,
    max_len: int = 4,
    num_data: int = -1,
    data_split: Optional[List[float]] = [0.8, 0.1, 0.1],
    oversample: float = None,
    cache: str = None,
    homogenize: bool = False,
    emb_dim: int = 100,
    temporal_closure: bool = False,
    motifs_to_use: List[int] = [0],
    motifs_dir: Optional[str] = None,
) -> Tuple[torchtext.vocab.Vocab, GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    if num_data > 0:
        num_train_data = math.floor(data_split[0] * num_data)
        num_dev_data = math.floor(data_split[1] * num_data)
        num_test_data = math.floor(data_split[2] * num_data)
    else:
        num_train_data = num_dev_data = num_test_data = (
            -1
        )  # when splits are predetermined

    task_name = task_names[0]

    def cachename(base: str):
        base += f"_os{str(oversample).split('.')[1]}" if oversample is not None else ""
        base += "_temporal_closure" if temporal_closure else ""
        return base + ".pt"

    train_cache = cachename(f"train_{num_train_data}_{task_name}")
    dev_cache = cachename(f"dev_{num_dev_data}_{task_name}")
    test_cache = cachename(f"test_{num_test_data}_{task_name}")
    vocab_cache = cachename(f"vocab_{num_train_data}_{task_name}")
    if cache is not None and os.path.isfile(os.path.join(cache, train_cache)):
        train = torch.load(os.path.join(cache, train_cache))

        fnames, labels, graphs = zip(*train)
        label_set = set(
            # [indiv_label for label_lst in labels for indiv_label in label_lst]
            [label.item() for label in labels]
        )
        label_dct = {label: i for i, label in enumerate(label_set)}

        def label_pipeline(x):
            return [label_dct[y.item()] for y in x]

        try:
            dev = torch.load(os.path.join(cache, dev_cache))
        except FileNotFoundError:
            dev = None
        try:
            test = torch.load(os.path.join(cache, test_cache))
        except FileNotFoundError:
            test = None
        vocab = torch.load(os.path.join(cache, vocab_cache))
        edge_vocab = {etype for graph in graphs for etype in graph.canonical_etypes}
        edge_vocab.update(doctime_etypes)
        edge_vocab.update(
            {(e2, category_reverse(rel), e1) for e1, rel, e2 in edge_vocab}
        )
    else:
        train, dev, test = load_data_dir(
            data_dir=data_dir,
            label_dir=labels_path,
            pretrained_dir=pretrained_dir,
            labels=task_names,
            num_train_data=num_train_data,
            num_dev_data=num_dev_data,
            num_test_data=num_test_data,
            oversample=oversample,
        )

        fnames, labels, graphs = train
        assert isinstance(labels[0], np.ndarray)
        vocab, edge_vocab, text_pipeline = get_vocab_and_pipeline(
            graphs, temporal_closure=temporal_closure
        )
        label_set = set(
            [indiv_label for label_lst in labels for indiv_label in label_lst]
        )
        label_dct = {label: i for i, label in enumerate(label_set)}

        def label_pipeline(x):
            return [label_dct[y[0]] for y in x]

        dgl_graphs = graphify(
            data=graphs,
            edge_vocab=edge_vocab,
            text_pipeline=text_pipeline,
            max_len=max_len,
            pretrained=pretrained_dir is not None,
            temporal_closure=temporal_closure,
        )
        train = list(zip(fnames, labels, dgl_graphs))

        fnames, labels, graphs = dev
        dgl_graphs = graphify(
            data=graphs,
            edge_vocab=edge_vocab,
            text_pipeline=text_pipeline,
            max_len=max_len,
            pretrained=pretrained_dir is not None,
            temporal_closure=temporal_closure,
        )
        dev = list(zip(fnames, labels, dgl_graphs))

        fnames, labels, graphs = test
        dgl_graphs = graphify(
            data=graphs,
            edge_vocab=edge_vocab,
            text_pipeline=text_pipeline,
            max_len=max_len,
            pretrained=pretrained_dir is not None,
            temporal_closure=temporal_closure,
        )
        test = list(zip(fnames, labels, dgl_graphs))

        if cache is not None:
            torch.save(train, os.path.join(cache, train_cache))
            torch.save(dev, os.path.join(cache, dev_cache))
            torch.save(test, os.path.join(cache, test_cache))
            torch.save(vocab, os.path.join(cache, vocab_cache))

    homogenize = (
        True if (len(motifs_to_use) > 0 and motifs_to_use != [0]) else homogenize
    )
    train_dataloader = get_loader(
        data=train,
        label_pipeline=label_pipeline,
        batch_size=batch_size,
        homogenize=homogenize,
        pretrained_dir=pretrained_dir,
        emb_dim=emb_dim,
        motifs_to_use=motifs_to_use,
        edge_vocab=edge_vocab,
        motifs_dir=motifs_dir,
    )
    dev_dataloader = get_loader(
        data=dev,
        label_pipeline=label_pipeline,
        batch_size=batch_size,
        homogenize=homogenize,
        pretrained_dir=pretrained_dir,
        emb_dim=emb_dim,
        motifs_to_use=motifs_to_use,
        edge_vocab=edge_vocab,
        motifs_dir=motifs_dir,
    )

    test_dataloader = get_loader(
        data=test,
        label_pipeline=label_pipeline,
        batch_size=batch_size,
        homogenize=homogenize,
        pretrained_dir=pretrained_dir,
        emb_dim=emb_dim,
        motifs_to_use=motifs_to_use,
        edge_vocab=edge_vocab,
        motifs_dir=motifs_dir,
    )

    return (
        vocab,
        edge_vocab,
        list(label_dct.keys()),
        train_dataloader,
        dev_dataloader,
        test_dataloader,
    )
