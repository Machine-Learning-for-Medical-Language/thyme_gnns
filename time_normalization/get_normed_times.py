import argparse
import json
import os
import pickle
import re
from py4j.java_gateway import JavaGateway, GatewayParameters, Py4JJavaError

gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
normer = gateway.entry_point.getTimeNormer()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g", "--graphs_dir", type=str, help="directory of original graph data"
)
parser.add_argument(
    "-o", "--out_dir", type=str, help="directory where output will be saved"
)
parser.add_argument("-t", "--doctime_path", type=str, help="path of doctime file")
parser.add_argument(
    "--preprocess",
    action="store_true",
    help="clean date strings so they can be properly normalized. keeping this off may lead to some timexes being omitted",
)
args = parser.parse_args()

date_pattern = re.compile("\d{1,2}(\/|\-)\d{1,2}(\/|\-)?(\d\d(\d\d)?)?")


def process(fname, doctime):
    y, m, d = doctime.split("-")
    doctime_dct = {"y": int(y), "m": int(m), "d": int(d)}
    with open(os.path.join(args.graphs_dir, fname), "rb") as f:
        data = pickle.load(f)
    out = {}
    for timex in data["timex"]:
        entry = data["timex"][timex]
        timex_text = entry["text"]
        if (
            timex_text.is_numeric()
            or timex_text.endswith("ly")
            or timex_text in ["the past", "the future"]
        ):
            continue
        try:
            if args.preprocess:
                match = re.search(date_pattern, timex_text)
                if match:
                    timex_text = match.group()
            normed = normer.norm(timex_text, doctime_dct)
            if normed.startswith("Period"):
                continue
            out[timex] = normed
        except Py4JJavaError:
            print(timex_text)
            continue
    # dump only the timexes that could be normed
    with open(os.path.join(args.out_dir, fname), "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    fnames = os.listdir(args.graphs_dir)
    with open(args.doctime_path, "r") as f:
        doctimes = json.load(f)
    # for doctime_dct in doctimes:
    #     process(doctime_dct["file_name"], doctime_dct["Discharge_date"])
    for i, fname in enumerate(fnames):
        process(fname, doctimes[i]["Discharge_date"])
