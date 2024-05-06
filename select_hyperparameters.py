#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import itertools
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--hyperparameters", type=str)
parser.add_argument("-i", "--indices", nargs="+")
parser.add_argument("--default_args", type=str)
args = parser.parse_args()


def get_job_from_hyperparameters(hyperparameters, indices):
    sorted_pairs = sorted(hyperparameters.items())

    paired_lists = [[(key, value) for value in values] for key, values in sorted_pairs]
    if len(indices) == 1:
        index = indices[0]
        cartesian_product = [dict(job) for job in itertools.product(*paired_lists)]
        return cartesian_product[index]
    elif len(indices) == len(paired_lists):
        return dict(pair[index] for pair, index in zip(paired_lists, indices))
    else:
        raise ValueError(
            f"Expected {len(paired_lists)} indices but received {len(indices)}"
        )


def main():
    """
    Produces the hyperparameter dictionary for a job index from a
    hyperparameter search space.

    When providing a job index, either supply a single number or supply
    a space-separated list of numbers indexing into the list of options
    for each hyperparameter, in alphabetical order.
    """
    # with hyperparameters.open('r', encoding='utf8') as hp_file:
    with open(args.hyperparameters, "r") as hp_file:
        hp_dict = json.load(hp_file)
    indices = [int(index) for index in args.indices]

    if args.default_args is not None:
        with open(args.default_args, "r", encoding="utf8") as def_args_file:
            default_args_dict = json.load(def_args_file)
        hyperparameters = {
            **default_args_dict,
            **get_job_from_hyperparameters(hp_dict, indices),
        }
    else:
        hyperparameters = get_job_from_hyperparameters(hp_dict, indices)

    # click.echo(json.dumps(hyperparameters, indent=2))
    sys.stdout.write(json.dumps(hyperparameters))


if __name__ == "__main__":
    main()
