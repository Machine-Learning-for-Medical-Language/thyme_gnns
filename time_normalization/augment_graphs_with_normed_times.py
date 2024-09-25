import argparse
import json
import os
import pdb
import pickle
from datetime import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g", "--graphs_dir", type=str, help="directory of original graph data"
)  # should be the version WITH embeddings, so we can remove them from normed_times for free
parser.add_argument(
    "-n", "--normed_times_dir", type=str, help="directory of normalized timexes"
)
parser.add_argument(
    "-o", "--output_dir", type=str, help="directory where output will be saved"
)
parser.add_argument("-t", "--doctime_path", type=str, help="path of doctime file")
args = parser.parse_args()


def augment(fname, doctime):
    with open(os.path.join(args.graphs_dir, fname), "rb") as f:
        input_graph = pickle.load(f)
    with open(os.path.join(args.normed_times_dir, fname), "rb") as f:
        normed_times = pickle.load(f)
    doctime = datetime.fromisoformat(doctime)
    output_graph = {"timex": {}, "normed_time": {}}
    output_graph["event"] = input_graph["event"]
    output_graph["relations"] = input_graph["relations"]
    # normalized times should be a separate ntype
    rels_to_ignore = (
        set()
    )  # keep track of relations that contain Timexes that have been transformed to Normed_Timexes
    for timex in input_graph["timex"]:
        if timex in normed_times:
            if (
                len(normed_times[timex]["rel_days"]) == 1
            ):  # some timexes have only one normed time (e.g. "12/5/19" -> [0.0] if 12/5/19 is the discharge date)
                normed_timex_str = (
                    f"Normed_Time_{len(output_graph['normed_time'])}"  # make id
                )
                output_graph["normed_time"][normed_timex_str] = {
                    "text": normed_times[timex]["text"],
                    "relative_time": normed_times[timex]["rel_days"][0],
                }
                # output_graph["normed_time"][normed_timex_str] = input_graph["timex"][timex]
                # output_graph["normed_time"][normed_timex_str]["relative_time"] = normed_times[timex]["rel_days"][0]
                for i, rel in enumerate(input_graph["relations"]):
                    if rel["arg1"] == timex:
                        rels_to_ignore.add(i)
                        output_graph["relations"].append(
                            {
                                "arg1": normed_timex_str,
                                "arg2": rel["arg2"],
                                "category": rel["category"],
                            }
                        )
                    elif rel["arg2"] == timex:
                        rels_to_ignore.add(i)
                        output_graph["relations"].append(
                            {
                                "arg1": rel["arg1"],
                                "arg2": normed_timex_str,
                                "category": rel["category"],
                            }
                        )
            else:  # some timexes have a start and an end (e.g. "the past two weeks")
                normed_timex_strs = [
                    f"Normed_Time_{len(output_graph['normed_time'])}",
                    f"Normed_Time_{len(output_graph['normed_time'])+1}",
                ]
                for i in range(2):
                    # output_graph["normed_time"][normed_timex_strs[i]] = input_graph["timex"][timex]
                    # output_graph["normed_time"][normed_timex_strs[i]]["relative_time"] = normed_times[timex]["rel_days"][i]
                    output_graph["normed_time"][normed_timex_strs[i]] = {
                        "text": normed_times[timex]["text"],
                        "relative_time": normed_times[timex]["rel_days"][0],
                    }
                for i, rel in enumerate(input_graph["relations"]):
                    if rel["arg1"] == timex or rel["arg2"] == timex:
                        rels_to_ignore.add(i)
                        if rel["category"] == "OVERLAP":
                            # these categories capture the uncertainty of OVERLAP, but require specific ordering
                            categories = ["BEGINS-ON", "ENDS-ON"]
                            if rel["arg1"] == timex:
                                event_str = rel["arg2"]
                            elif rel["arg2"] == timex:
                                event_str = rel["arg1"]
                            for j in range(2):
                                output_graph["relations"].append(
                                    {
                                        "arg1": event_str,
                                        "arg2": normed_timex_strs[j],
                                        "category": categories[j],
                                    }
                                )
                        elif rel["category"] in ["BEFORE", "CONTAINS", "NOTED-ON"]:
                            categories = [rel["category"]] * 2
                            if rel["arg1"] == timex:
                                for j in range(2):
                                    output_graph["relations"].append(
                                        {
                                            "arg1": normed_timex_strs[j],
                                            "arg2": rel["arg2"],
                                            "category": categories[j],
                                        }
                                    )
                            elif rel["arg2"] == timex:
                                for j in range(2):
                                    output_graph["relations"].append(
                                        {
                                            "arg1": rel["arg1"],
                                            "arg2": normed_timex_strs[j],
                                            "category": categories[j],
                                        }
                                    )
                        elif (
                            rel["category"] == "BEGINS-ON"
                        ):  # event was noted to begin on or after the timex; use first timestamp
                            if rel["arg1"] == timex:
                                output_graph["relations"].append(
                                    {
                                        "arg1": normed_timex_strs[0],
                                        "arg2": rel["arg2"],
                                        "category": "BEGINS-ON",
                                    }
                                )
                            elif rel["arg2"] == timex:
                                output_graph["relations"].append(
                                    {
                                        "arg1": rel["arg1"],
                                        "arg2": normed_timex_strs[0],
                                        "category": "BEGINS-ON",
                                    }
                                )
                        elif rel["category"] == "ENDS-ON":  # use second timestamp
                            if rel["arg1"] == timex:
                                output_graph["relations"].append(
                                    {
                                        "arg1": normed_timex_strs[0],
                                        "arg2": rel["arg2"],
                                        "category": "ENDS-ON",
                                    }
                                )
                            elif rel["arg2"] == timex:
                                output_graph["relations"].append(
                                    {
                                        "arg1": rel["arg1"],
                                        "arg2": normed_timex_strs[0],
                                        "category": "ENDS-ON",
                                    }
                                )
                        # timexes that are omitted from the above will still get used in timex-timex tlinks... will this be helpful? should i remove them?
                        else:
                            pdb.set_trace()
                            # print(rel["category"])
        else:
            output_graph["timex"][timex] = input_graph["timex"][timex]

    # remove only the rels involving timexes that've been removed
    output_graph["relations"] = [
        rel
        for i, rel in enumerate(output_graph["relations"])
        if i not in rels_to_ignore
    ]
    for rel in output_graph[
        "relations"
    ]:  # make sure we don't have any stragglers --- rels containing timexes that we've normed and removed
        if rel["arg1"].startswith("Timex"):
            assert rel["arg1"] in output_graph["timex"]
        elif rel["arg2"].startswith("Timex"):
            assert rel["arg2"] in output_graph["timex"]

    # add relations between timexes
    normed_timex_keys = list(output_graph["normed_time"].keys())
    for i in range(len(normed_timex_keys)):
        for j in range(i + 1, len(normed_timex_keys)):
            normed_time_0 = output_graph["normed_time"][normed_timex_keys[i]][
                "relative_time"
            ]
            normed_time_1 = output_graph["normed_time"][normed_timex_keys[j]][
                "relative_time"
            ]
            if normed_time_0 < normed_time_1:
                output_graph["relations"].append(
                    {
                        "arg1": normed_timex_keys[i],
                        "arg2": normed_timex_keys[j],
                        "category": "BEFORE",
                    }
                )
            elif normed_time_0 > normed_time_1:
                output_graph["relations"].append(
                    {
                        "arg1": normed_timex_keys[j],
                        "arg2": normed_timex_keys[i],
                        "category": "BEFORE",
                    }
                )
            else:
                output_graph["relations"].append(
                    {
                        "arg1": normed_timex_keys[i],
                        "arg2": normed_timex_keys[j],
                        "category": "OVERLAP",
                    }
                )

    # add doctimerel for timexes
    # doctime = doctimes[fname]
    for timex in normed_timex_keys:
        normed_time = output_graph["normed_time"][timex]["relative_time"]
        # if normed_time < 0:
        #     output_graph["normed_time"][timex]["dtr"] = "BEFORE"
        # elif normed_time > 0:
        #     output_graph["normed_time"][timex]["dtr"] = "AFTER"
        # else:
        #     output_graph["normed_time"][timex]["dtr"] = "OVERLAP"
        if normed_time < -365:
            output_graph["normed_time"][timex]["dtr"] = "ANCIENT-HISTORY"
        elif normed_time < 0:
            output_graph["normed_time"][timex]["dtr"] = "RECENT-HISTORY"
        elif normed_time > 0:
            output_graph["normed_time"][timex]["dtr"] = "AFTER"
        else:
            output_graph["normed_time"][timex]["dtr"] = "OVERLAP"

    with open(os.path.join(args.output_dir, fname), "wb") as f:
        pickle.dump(output_graph, f)


if __name__ == "__main__":
    fnames = os.listdir(args.graphs_dir)
    with open(args.doctime_path, "r") as f:
        doctimes = json.load(f)
    for doctime_dct in tqdm(doctimes):
        fname = doctime_dct["file_name"].replace("json", "pkl")
        if os.path.isfile(os.path.join(args.graphs_dir, fname)) and os.path.isfile(
            os.path.join(args.normed_times_dir, fname)
        ):
            augment(fname, doctime_dct["Discharge_date"])
