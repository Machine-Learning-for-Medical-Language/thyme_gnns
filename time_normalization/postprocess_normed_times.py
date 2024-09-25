import argparse
import json
import os
import pickle
import re
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--normed_times_dir", type=str, help="directory of normalized timexes"
)
parser.add_argument(
    "-o", "--out_dir", type=str, help="directory where output will be saved"
)
parser.add_argument("-t", "--doctime_path", type=str, help="path of doctime file")
args = parser.parse_args()

iso_pattern = re.compile("\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def rel_days(anchor, rel_date):
    rel_date = datetime.fromisoformat(rel_date)
    return (anchor - rel_date).seconds / 86400


def process(fname, doctime):
    fname = fname.replace("json", "pkl")
    with open(os.path.join(args.normed_times_dir, fname), "rb") as f:
        normed_times = pickle.load(f)
    doctime = datetime.fromisoformat(doctime)
    out = {}
    for id in normed_times:
        entry = normed_times[id]
        if not entry["normed_time"].startswith("TimeSpan"):
            continue
        endpoints = re.findall(iso_pattern, entry["normed_time"])
        assert len(endpoints) == 2
        rel_beginning, rel_end = map(lambda x: rel_days(x, doctime), endpoints)
        if rel_end - rel_beginning <= 1:  # <= 1 day; 1 node
            out[id] = {
                "text": entry["text"],
                "rel_days": [sum(rel_beginning, rel_end) / 2],
            }
        else:
            out[id] = {"text": entry["text"], "rel_days": [rel_beginning, rel_end]}
    with open(os.path.join(args.out_dir, fname), "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    fnames = os.listdir(args.normed_times_dir)
    with open(args.doctime_path, "r") as f:
        doctimes = json.load(f)
    for doctime_dct in doctimes:
        process(doctime_dct["file_name"], doctime_dct["Discharge_date"])
