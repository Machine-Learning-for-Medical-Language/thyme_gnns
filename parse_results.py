import argparse
from datetime import datetime as dt
import json
import numpy as np
import os
import pandas as pd
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--results_dir", type=str)
parser.add_argument("-j", "--job", nargs="+")
parser.add_argument("--full_stats", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    full = {}
    for _job in args.job:
        job_dir = os.path.join(args.results_dir, _job)
        for run_dir in os.listdir(job_dir):
            if not os.path.isdir(os.path.join(job_dir, run_dir)):
                continue
            this_run = {}
            with open(os.path.join(job_dir, run_dir, "config.json"), "r") as f:
                config = json.load(f)
            this_run.update(config)
            with open(os.path.join(job_dir, run_dir, "train.log"), "r") as f:
                log = f.readlines()
            epoch_blocks, train_blocks, dev_blocks = [], [], []
            in_train_block, in_dev_block = False, False
            for line in log:
                if "Epoch" in line:
                    if in_dev_block:
                        dev_blocks.append(this_block)
                        this_block = []
                    epoch_blocks.append([line])
                elif "Training loss" in line:
                    in_train_block = True
                    this_block = [line]
                elif "Dev loss" in line:
                    if in_train_block:
                        train_blocks.append(this_block)
                    in_dev_block = True
                    this_block = [line]
                elif in_train_block or in_dev_block:
                    this_block.append(line)
                elif "Loading" in line:
                    continue
                else:
                    print("this should not happen!")
                    break
            # get dev minority f1s
            dev_min_f1s = [float(block[-1].split()[-2]) for block in dev_blocks]
            if len(dev_min_f1s) == 0:
                continue
            this_run["epochs_completed"] = len(dev_blocks)
            peak_dev_min_f1 = (max(dev_min_f1s), np.argmax(dev_min_f1s) + 1)
            num_epochs_post_peak = len(epoch_blocks) - peak_dev_min_f1[1]
            if args.full_stats:
                this_run["peak_epoch"] = peak_dev_min_f1[1] - 1
                peak_dev = dev_blocks[this_run["peak_epoch"]]
                stat_lines = peak_dev[-2:]
                for line in stat_lines:
                    stats = line.split()
                    category = stats[0]
                    this_run["accuracy"] = stats[1]
                    this_run[f"precision_{category}"] = stats[2]
                    this_run[f"recall_{category}"] = stats[3]
                    this_run[f"f1_{category}"] = stats[4]
                this_run
            else:
                this_run["peak_min_f1"], this_run["peak_epoch"] = peak_dev_min_f1

            # get standard deviation in dev minority f1s after peak
            this_run["post_peak_f1_stdev"] = (
                np.std(dev_min_f1s[peak_dev_min_f1[1] :])
                if num_epochs_post_peak > 1
                else 0
            )

            # get loss at peak dev, final epoch
            model_losses = [float(line[0].split()[-1]) for line in epoch_blocks]
            loss_at_peak = model_losses[peak_dev_min_f1[-1] - 1]
            loss_at_end = model_losses[-1]
            loss_difference = loss_at_end - loss_at_peak
            this_run["loss_difference"] = loss_difference
            this_run["loss_difference/remaining_epochs"] = (
                loss_difference / num_epochs_post_peak
                if num_epochs_post_peak > 0
                else 0
            )

            # get epoch training time
            last_dev_time = " ".join(dev_blocks[-2][1].split()[:2])
            last_dev_dt = dt.strptime(last_dev_time, "%Y-%m-%d %H:%M:%S,%f")
            last_epoch_time = " ".join(epoch_blocks[-1][0].split()[:2])
            last_epoch_dt = dt.strptime(last_epoch_time, "%Y-%m-%d %H:%M:%S,%f")
            td = last_epoch_dt - last_dev_dt
            this_run["train_epoch_time"] = last_epoch_dt - last_dev_dt
            full[f"{_job}_{run_dir}"] = this_run
    full_df = pd.DataFrame.from_dict(full, orient="index")
    for col in full_df.columns:
        if full_df[col].dtype == list:
            full_df[col] = full_df[col].astype(str)
    cols_to_keep = [col for col in full_df.columns if len(full_df[col].unique()) > 1]
    full_df = full_df[cols_to_keep]
    if args.full_stats:
        full_df.to_csv(
            os.path.join(
                args.results_dir, args.job[-1], f"{'_'.join(args.job)}_full.csv"
            ),
            index=False,
        )
    else:
        full_df.to_csv(
            os.path.join(args.results_dir, args.job[-1], f"{'_'.join(args.job)}.csv"),
            index=False,
        )
