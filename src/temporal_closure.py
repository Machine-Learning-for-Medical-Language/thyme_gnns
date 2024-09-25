import argparse
import dgl
import os
import pdb
import pickle
import torch
from tqdm import tqdm
from typing import Dict, Union, Optional, Any, Tuple, List

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, help="directory of data")
args = parser.parse_args()

str_to_i = {
    "ENDS-ON": 0,
    "ENDS-ON-1": 1,
    "BEFORE": 2,
    "CONTAINS": 3,
    "CONTAINS-1": 4,
    "CONTAINS-SUBEVENT": 5,
    "CONTAINS-SUBEVENT-1": 6,
    "NOTED-ON": 7,
    "NOTED-ON-1": 8,
    "BEGINS-ON": 9,
    "BEGINS-ON-1": 10,
    "AFTER": 11,
    "OVERLAP": 12,
    "BEFORE/OVERLAP": 13,
}

i_to_str = {item: key for key, item in str_to_i.items()}


def reverse_relation(rel: str) -> str:
    if rel == "OVERLAP":
        return "OVERLAP"
    elif rel == "BEFORE":
        return "AFTER"
    elif rel == "AFTER":
        return "BEFORE"
    elif rel.endswith("-1"):
        return rel[:-2]
    else:
        return rel + "-1"


rel_rel_extender = {
    "ab-bc": {
        "ENDS-ON": {  # if a-b is type x and b-c is type y, then...
            "BEFORE": "BEFORE",  # a ends-on b, b before c ==> a before c
            "ENDS-ON-1": "OVERLAP",
            "CONTAINS-1": "OVERLAP",
            "BEGINS-ON-1": "OVERLAP",
        },
        "BEFORE": {
            "ENDS-ON": "BEFORE",
            "BEFORE": "BEFORE",
            "CONTAINS": "BEFORE",
            "BEGINS-ON-1": "BEFORE",
        },
        "OVERLAP": {
            "ENDS-ON-1": "OVERLAP",
            "CONTAINS-1": "OVERLAP",
            "BEGINS-ON-1": "OVERLAP",
        },
        "CONTAINS": {
            "ENDS-ON": "OVERLAP",
            "OVERLAP": "OVERLAP",
            "CONTAINS": "CONTAINS",
            "BEGINS-ON": "CONTAINS",
            "ENDS-ON-1": "OVERLAP",
            "OVERLAP-1": "OVERLAP",
            "CONTAINS-1": "OVERLAP",
            "BEGINS-ON-1": "OVERLAP",
        },
    },
    "ab-ac": {
        "ENDS-ON": {  # a ends-on b, a ends-on c => b overlap c
            "BEGINS-ON": "AFTER",
            "CONTAINS": "OVERLAP",
        },
        "BEFORE": {
            "CONTAINS": "AFTER",
            "ENDS-ON-1": "AFTER",
            "AFTER": "AFTER",
        },
        "OVERLAP": {"CONTAINS-1": "OVERLAP"},
        "CONTAINS": {
            "BEFORE": "BEFORE",
            "AFTER": "AFTER",
        },
    },
}

dtr_rel_extender = {
    "BEFORE": {  # a before dt
        "CONTAINS": "BEFORE",  # a _ b => b _ dt
        "ENDS-ON-1": "BEFORE",
        "AFTER": "BEFORE",
    },
    "BEFORE/OVERLAP": {
        "BEGINS-ON": "BEFORE",
        "AFTER": "BEFORE",
    },
    "OVERLAP": {
        "CONTAINS-1": "OVERLAP",
    },
    "AFTER": {"ENDS-ON": "AFTER", "BEFORE": "AFTER", "BEGINS-ON-1": "AFTER"},
}

dtr_dtr_extender = {
    "BEFORE": {
        "AFTER": "BEFORE",
    },  # a before dt, b after dt ==> a before b
    "BEFORE/OVERLAP": {
        "BEFORE/OVERLAP": "OVERLAP",
        "OVERLAP": "OVERLAP",
    },
    "OVERLAP": {
        "BEFORE/OVERLAP": "OVERLAP",
        "OVERLAP": "OVERLAP",
    },
    "AFTER": {
        "BEFORE": "AFTER",
    },
}

DTR_TYPES = ["BEFORE", "BEFORE/OVERLAP", "OVERLAP", "AFTER"]
stricter_than_overlap = [
    "BEFORE/OVERLAP",
    "CONTAINS",
    "CONTAINS-SUBEVENT",
    "NOTED-ON",
    "BEGINS-ON",
    "ENDS-ON",
]

potential_edge_types = []
for nt0 in ["event", "timex"]:
    for et in str_to_i:
        for nt1 in ["event", "timex"]:
            new_edge_type = (nt0, et, nt1)
            if new_edge_type not in potential_edge_types:
                potential_edge_types.append(new_edge_type)


# data = dictionary with entities and relations
def extrapolate(data: Dict[str, Dict]) -> Dict[str, Dict]:
    e_to_i = {"event": {}, "timex": {}}
    i_to_e = {"event": {}, "timex": {}}
    for i, event in enumerate(data["event"]):
        e_to_i["event"][event] = i
        i_to_e["event"][i] = event
    # for i, timex in enumerate(data["timex"]):  # below, we only include timexes that are actually used
    #     e_to_i["timex"][timex] = i
    #     i_to_e["timex"][i] = timex

    def which_is_gold(
        pair1: torch.Tensor,
        pair2: torch.Tensor,
        et1: Tuple[str],
        et2: Tuple[str],
    ) -> int:
        rel1_arg1_ntype, rel1_rel, rel1_arg2_ntype = et1
        rel1_arg1 = i_to_e[rel1_arg1_ntype][int(pair1[0])]
        rel1_arg2 = i_to_e[rel1_arg2_ntype][int(pair1[1])]
        reverse_rel1_rel = reverse_relation(rel1_rel)
        rel1_data = [
            rel
            for rel in data["relations"]
            if rel["arg1"] == rel1_arg1
            and rel["arg2"] == rel1_arg2
            and rel["category"] == rel1_rel
        ]
        rel1_data.extend(
            [
                rel
                for rel in data["relations"]
                if rel["arg1"] == rel1_arg2
                and rel["arg2"] == rel1_arg1
                and rel["category"] == reverse_rel1_rel
            ]
        )

        rel2_arg1_ntype, rel2_rel, rel2_arg2_ntype = et2
        rel2_arg1 = i_to_e[rel2_arg1_ntype][int(pair2[0])]
        rel2_arg2 = i_to_e[rel2_arg2_ntype][int(pair2[1])]
        reverse_rel2_rel = reverse_relation(rel2_rel)
        rel2_data = [
            rel
            for rel in data["relations"]
            if rel["arg1"] == rel2_arg1
            and rel["arg2"] == rel2_arg2
            and rel["category"] == rel2_rel
        ]
        rel2_data.extend(
            [
                rel
                for rel in data["relations"]
                if rel["arg1"] == rel2_arg2
                and rel["arg2"] == rel2_arg1
                and rel["category"] == reverse_rel2_rel
            ]
        )
        if len(rel1_data) > 0 and len(rel2_data) == 0:
            return 1
        elif len(rel2_data) > 0 and len(rel1_data) == 0:
            return 2
        if et1 == et2 and len(rel1_data) > 0 and len(rel2_data) > 0:
            pdb.set_trace()
        return 0

    def check_non_contradictory(edges: Dict[str, Dict]) -> Dict[str, Dict]:
        for et in edges:
            if et[1] != "OVERLAP":
                overlap_edges = edges[(et[0], "OVERLAP", et[2])].t()
                # pair and inverse shouldn't appear in the same category
                pairs = edges[et].t()
                reverse_pairs = pairs.flip(dims=[1])
                for pair in pairs:
                    # check for cases where e.g. event 0 and doctime 0 are paired
                    if pair[0] != pair[1] and torch.any(
                        torch.all(reverse_pairs == pair, dim=1)
                    ):
                        gold = which_is_gold(
                            pair1=pair, pair2=pair.flip(dims=[0]), et1=et, et2=et
                        )
                        if gold == 1:
                            # first version is present in original data; remove second
                            edges[et] = edges[et][
                                (edges[et] != pair.flip(dims=[0]).unsqueeze(1)).any(
                                    dim=1
                                )
                            ]
                        elif gold == 2:
                            # other way around
                            edges[et] = edges[et][
                                (edges[et] != pair.unsqueeze(1)).any(dim=1)
                            ]
                        else:
                            # neither or both are gold ==> both should be removed
                            # replace with "overlap", since *some* connection is consistently predicted
                            if "timex" in et:
                                print("pdb should be tracing right now!")
                                pdb.set_trace()
                            edges[et] = pairs[(pairs != pair).any(dim=1)].t()
                            if not (overlap_edges == pair).all(dim=1).any():
                                edges = add_new_edges(
                                    edges=edges,
                                    et=(et[0], "OVERLAP", et[2]),
                                    new_srcs_dsts=pair.unsqueeze(1),
                                )
            # pair should not be in the tensors for multiple edge types
            for et1 in edges:
                if et[0] == et1[0] and et[1] != et1[1] and et[2] == et1[2]:
                    pairs = edges[et].t()
                    pairs1 = edges[et1].t()
                    for pair in pairs:
                        if torch.any(torch.all(pairs1 == pair, dim=1)):
                            # if one of the categories is "overlap", go with the more specific label
                            if (
                                et[1] == "OVERLAP"
                                and et1[1].replace("-1", "") in stricter_than_overlap
                            ):
                                edges[et] = pairs[(pairs != pair).any(dim=1)].t()
                            elif (
                                et1[1] == "OVERLAP"
                                and et[1].replace("-1", "") in stricter_than_overlap
                            ):
                                edges[et1] = pairs1[(pairs1 != pair).any(dim=1)].t()
                            elif (et[0] == "timex" and et[2] == "DocTime") or (
                                et[0] == "DocTime" and et[2] == "timex"
                            ):
                                # no gold => can't verify => remove both
                                edges[et] = pairs[(pairs != pair).any(dim=1)].t()
                                edges[et1] = pairs1[(pairs1 != pair).any(dim=1)].t()
                            else:  # labels are directly contradictory (e.g. "CONTAINS" and "BEFORE")
                                # check for and keep "gold" relation (i.e. present in the original predictions)
                                gold = which_is_gold(
                                    pair1=pair, pair2=pair, et1=et, et2=et1
                                )
                                if gold == 1:
                                    # first is legitimate; keep it
                                    edges[et1] = pairs1[(pairs1 != pair).any(dim=1)].t()
                                elif gold == 2:
                                    # second is legitimate
                                    edges[et] = pairs[(pairs != pair).any(dim=1)].t()
                                else:
                                    # if neither is "gold", or if both are (=false prediction), remove both
                                    edges[et] = pairs[(pairs != pair).any(dim=1)].t()
                                    edges[et1] = pairs1[(pairs1 != pair).any(dim=1)].t()
        return edges

    # populate edges from given data
    edges = {et: torch.tensor([[], []]) for et in potential_edge_types}
    # shape of edges[whatever] will be 2 x n
    for rel in data["relations"]:
        arg1_etype = "timex" if rel["arg1"].startswith("Timex") else "event"
        arg2_etype = "timex" if rel["arg2"].startswith("Timex") else "event"
        et = rel["category"]
        # check for contradictory relations
        contradictory = False
        for rel1 in data["relations"]:
            # x REL y and y REL x
            if (
                et != "OVERLAP"
                and rel1["category"] == et
                and rel1["arg1"] == rel["arg2"]
                and rel1["arg2"] == rel["arg1"]
            ):
                contradictory = True
            # same pair in two different categories
            elif rel1["category"] != rel["category"]:
                if rel1["arg1"] == rel["arg1"] and rel1["arg2"] == rel["arg2"]:
                    contradictory = True
                elif (
                    rel1["arg2"] == rel["arg1"]
                    and rel1["arg1"] == rel["arg2"]
                    and rel1["category"] != reverse_relation(et)
                ):
                    contradictory = True
        if contradictory:
            continue
        if arg1_etype == "timex":  # catalog the timexes that we actually use
            if rel["arg1"] not in e_to_i["timex"]:
                e_to_i["timex"][rel["arg1"]] = len(e_to_i["timex"])
                i_to_e["timex"][e_to_i["timex"][rel["arg1"]]] = rel["arg1"]
        if arg2_etype == "timex":
            if rel["arg2"] not in e_to_i["timex"]:
                e_to_i["timex"][rel["arg2"]] = len(e_to_i["timex"])
                i_to_e["timex"][e_to_i["timex"][rel["arg2"]]] = rel["arg2"]
        arg1_i = e_to_i[arg1_etype][rel["arg1"]]
        arg2_i = e_to_i[arg2_etype][rel["arg2"]]
        # edges stored in the form: {(ntype, etype, ntype): [sources, destinations]}
        edges[(arg1_etype, et, arg2_etype)] = torch.cat(
            (edges[(arg1_etype, et, arg2_etype)], torch.tensor([[arg1_i], [arg2_i]])),
            dim=1,
        )

        # make bidirectional
        reverse_et = reverse_relation(et)
        edges[(arg2_etype, reverse_et, arg1_etype)] = torch.cat(
            (
                edges[(arg2_etype, reverse_et, arg1_etype)],
                torch.tensor([[arg2_i], [arg1_i]]),
            ),
            dim=1,
        )
    for et in edges:
        edges[et] = edges[et].view(2, -1).to(int)
    # edge-edge extrapolation
    for et_ab in edges:  # et = edge type
        if len(edges[et_ab][0]) == 0:
            continue
        rel_ab = et_ab[1]
        rel_ab = (
            "CONTAINS" if rel_ab in ["CONTAINS-SUBEVENT", "NOTED-ON"] else rel_ab
        )  # these are essentially subtypes of "contains"
        if rel_ab in rel_rel_extender["ab-bc"]:
            for et_bc in edges:  # get all possible extensions of et_ab
                if len(edges[et_bc][0]) == 0:
                    continue
                # if et_ab[0] == "DocTime" and et_bc[-1] == "DocTime":
                #     continue
                if et_ab[2] != et_bc[0]:
                    continue
                rel_bc = et_bc[1]
                rel_bc = (
                    "CONTAINS"
                    if rel_bc in ["CONTAINS-SUBEVENT", "NOTED-ON"]
                    else rel_bc
                )  # these are effectively the same
                if rel_bc in rel_rel_extender["ab-bc"][rel_ab]:
                    rel_ac = rel_rel_extender["ab-bc"][rel_ab][rel_bc]
                    et_ac = (et_ab[0], rel_ac, et_bc[-1])
                    new_srcs_dsts = torch.tensor([[]])
                    for _b in edges[et_ab][1]:
                        _a = set(edges[et_ab][0][edges[et_ab][1] == _b].tolist())
                        _c = set(edges[et_bc][1][edges[et_bc][0] == _b].tolist())
                        # get cartesian product of as and cs
                        if len(_c) > 0:  # we know len(_a) > 0
                            _a = torch.tensor(list(_a)).to(int)
                            _c = torch.tensor(list(_c)).to(int)
                            if len(new_srcs_dsts[0]) == 0:
                                new_srcs_dsts = torch.cartesian_prod(_a, _c).t()
                            else:
                                new_srcs_dsts = torch.cat(
                                    (new_srcs_dsts, torch.cartesian_prod(_a, _c).t()),
                                    dim=1,
                                )
                    if len(new_srcs_dsts[0]) > 0:
                        new_srcs_dsts = new_srcs_dsts[
                            (new_srcs_dsts[0] != new_srcs_dsts[1]).repeat(2, 1)
                        ].view(
                            2, -1
                        )  # remove self-loops
                        edges = add_new_edges(
                            edges=edges, et=et_ac, new_srcs_dsts=new_srcs_dsts
                        )
        if rel_ab in rel_rel_extender["ab-ac"]:
            for (
                et_ac
            ) in (
                edges
            ):  # this is functionally the same as above; changing var name for clarity
                if len(edges[et_ac][0]) == 0:
                    continue
                # if et_ab[0] == "DocTime" and et_ac[-1] == "DocTime":
                #     continue
                if et_ab[0] != et_ac[0]:
                    continue
                rel_ac = et_ac[1]
                rel_ac = (
                    "CONTAINS"
                    if rel_ac in ["CONTAINS-SUBEVENT", "NOTED-ON"]
                    else rel_ac
                )  # effectively the same
                if rel_ac in rel_rel_extender["ab-ac"][rel_ab]:
                    rel_bc = rel_rel_extender["ab-ac"][rel_ab][rel_ac]
                    et_bc = (et_ab[-1], rel_bc, et_ac[-1])
                    new_srcs_dsts = torch.tensor([[]])
                    for _a in edges[et_ab][0]:
                        _b = set(edges[et_ab][1][edges[et_ab][0] == _a].tolist())
                        _c = set(edges[et_ac][1][edges[et_ac][0] == _a].tolist())
                        # get cartesian product of bs and cs
                        if len(_b) > 0 and len(_c) > 0:
                            _b = torch.tensor(list(_b)).to(int)
                            _c = torch.tensor(list(_c)).to(int)
                            if _b.shape == _c.shape and (_b == _c).all():
                                continue
                            if len(new_srcs_dsts[0]) == 0:
                                new_srcs_dsts = torch.cartesian_prod(_b, _c).t()
                            else:
                                new_srcs_dsts = torch.cat(
                                    (new_srcs_dsts, torch.cartesian_prod(_b, _c).t()),
                                    dim=1,
                                )
                    if len(new_srcs_dsts[0]) > 0:
                        new_srcs_dsts = new_srcs_dsts[
                            (new_srcs_dsts[0] != new_srcs_dsts[1]).repeat(2, 1)
                        ].view(
                            2, -1
                        )  # remove self-loops
                        edges = add_new_edges(
                            edges=edges, et=et_bc, new_srcs_dsts=new_srcs_dsts
                        )
    # augment `edges` to include doctimerels
    for nt in ["event", "timex"]:
        for et in DTR_TYPES:
            potential_edge_types.append((nt, et, "DocTime"))
            reverse_et = reverse_relation(et)
            potential_edge_types.append(("DocTime", reverse_et, nt))
            edges[(nt, et, "DocTime")] = torch.tensor([[], []])
            edges[("DocTime", reverse_et, nt)] = torch.tensor([[], []])
    # document the doctimerel for each event
    for event in data["event"]:
        event_i = e_to_i["event"][event]
        entry = data["event"][event]
        edges = add_new_edges(
            edges=edges,
            et=("event", entry["dtr"], "DocTime"),
            new_srcs_dsts=torch.tensor([[event_i], [0]]),
        )
    # extrapolate relations based on doctimerel
    for dtr0 in DTR_TYPES:
        _a = set(edges[("event", dtr0, "DocTime")][0].tolist())  # get possible sources
        if len(_a) == 0:
            continue
        # dtr + edge => get dtr for timexes
        for et in dtr_rel_extender[dtr0]:
            _b = set(edges[("event", et, "timex")][1].tolist())
            if len(_b) == 0:
                continue
            new_srcs_dsts = torch.cartesian_prod(
                torch.tensor(list(_b)), torch.tensor([0])
            ).t()
            if len(new_srcs_dsts) > 0:
                new_srcs_dsts = new_srcs_dsts[
                    (new_srcs_dsts[0] != new_srcs_dsts[1]).repeat(2, 1)
                ].view(
                    2, -1
                )  # remove self-loops
                dtr_timex = dtr_rel_extender[dtr0][et]
                edges = add_new_edges(
                    edges=edges,
                    et=("timex", dtr_timex, "DocTime"),
                    new_srcs_dsts=new_srcs_dsts,
                )
        # dtr + dtr to relation
        for dtr1 in DTR_TYPES:
            if dtr1 in dtr_dtr_extender[dtr0]:
                _b = set(edges[("event", dtr1, "DocTime")][0].tolist())
                rel_ab = dtr_dtr_extender[dtr0][dtr1]
                new_srcs_dsts = torch.cartesian_prod(
                    torch.tensor(list(_a)).to(int), torch.tensor([0])
                ).t()
                if len(new_srcs_dsts) > 0:
                    new_srcs_dsts = new_srcs_dsts[
                        (new_srcs_dsts[0] != new_srcs_dsts[1]).repeat(2, 1)
                    ].view(
                        2, -1
                    )  # remove self-loops
                    edges = add_new_edges(
                        edges=edges,
                        et=("event", rel_ab, "DocTime"),
                        new_srcs_dsts=new_srcs_dsts,
                    )
    for et in edges:
        if edges[et].shape[-1] > 0:
            edges[et] = edges[et].unique(dim=1)  # remove duplicate pairs
            if (
                et[0] == "DocTime"
            ):  # make sure doctime only has idx 0; otherwise, have mixed it up with some other node
                assert torch.all(edges[et][0] == 0)
            elif et[0] == "timex":  # make sure timex indices make sense
                assert torch.max(edges[et][0] < len(data["timex"]))
            if et[2] == "DocTime":
                assert torch.all(edges[et][1] == 0)
            elif et[2] == "timex":
                assert torch.max(edges[et][1] < len(data["timex"]))
    edges = check_non_contradictory(edges)
    return {
        et: edges[et] for et in edges if edges[et].shape[-1] > 0
    }, e_to_i  # remove edge types with no edges


def add_new_edges(
    edges: Dict[Tuple[str], torch.Tensor], et: str, new_srcs_dsts=torch.Tensor
) -> Dict[Tuple[str], torch.Tensor]:
    reverse_et = (et[-1], reverse_relation(et[1]), et[0])
    edges[et] = torch.cat((edges[et], new_srcs_dsts), axis=1)
    edges[reverse_et] = torch.cat(
        (edges[reverse_et], new_srcs_dsts.flip(dims=[0])), axis=1
    )
    return edges


if __name__ == "__main__":
    # load graphs
    fnames = os.listdir(args.input_dir)
    for fname in tqdm(fnames[:100]):
        with open(os.path.join(args.input_dir, fname), "rb") as f:
            data = pickle.load(f)
        edges = extrapolate(data)
