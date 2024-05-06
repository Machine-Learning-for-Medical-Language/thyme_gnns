import argparse
import ast
from datetime import datetime
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pandas as pd
import pdb
import sys

os.environ["DGLBACKEND"] = "pytorch"
import dgl
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Any, List, Set, Dict, Tuple, Union, Optional, Callable

from get_data import get_data_loaders
from model import (
    GCNClassifier,
    EGATClassifier,
    HGTClassifier,
    GATClassifier,
    MotifClassifier,
)
from utils import logging_config, get_device

parser = argparse.ArgumentParser()

# data arguments
parser.add_argument("-d", "--data_dir", type=str, help="directory of data")
parser.add_argument(
    "-l", "--label_path", type=str, help="directory where labels are stored"
)
parser.add_argument(
    "--pretrained_dir",
    type=str,
    help="directory where pretrained node embeddings are stored.",
)
parser.add_argument(
    "-s", "--save_dir", type=str, help="path where loaded data will be saved"
)
parser.add_argument(
    "-c",
    "--cache",
    default=None,
    help="cache to load data from, if available; directory to save data in if not.",
)
parser.add_argument(
    "--pretrained_embmat_dir",
    type=str,
    help="path where pretrained embedding layer is stored",
)

parser.add_argument(
    "-m", 
    "--motifs_dir",
    type=str,
    default=None,
    help="directory where motif adjacency matrices are stored"
)

# data/model arguments
parser.add_argument(
    "-t", "--task_names", nargs="+", help="data columns to use as labels"
)
parser.add_argument(
    "--start_epoch",
    type=int,
    default=0,
    help="index of first epoch. useful when picking up partially-finished training.",
)
parser.add_argument(
    "--hyperparameters",
    type=str,
    default="stdout",
    help='path to hyperparameter file. "stdout" (default) to read from stdout',
)
parser.add_argument(
    "--save_model", action="store_true", help="whether to save the best model"
)
parser.add_argument(
    "--delete_model",
    action="store_true",
    help="delete model after testing. this allows the user to load the best model for training, without taking up extra storage space.",
)
args = parser.parse_args()


def get_model(
    hyperparameters: Dict[str, Any],
    model_args: Dict[str, Any],
) -> Callable:
    match hyperparameters["model_type"]:
        case "GCN":
            return GCNClassifier(**model_args)
        case "HGT":
            return HGTClassifier(**model_args)
        case "GAT":
            return GATClassifier(**model_args)
        case "EGAT":
            return EGATClassifier(**model_args)
        case "Motif":
            return MotifClassifier(**model_args)


def get_model_args(
    vocab_size: int,
    labels: Union[List, Set],
    hyperparameters: Dict[str, Any],
    num_classes: int = 2,
) -> Dict[str, Any]:
    pretrained = args.pretrained_dir is not None
    model_args = {
        "pretrained": pretrained,
        "node_vocab_size": vocab_size,
        "edge_vocab": labels,
        "num_classes": num_classes,
        "pretrained_embmat_dir": args.pretrained_embmat_dir,
    }
    model_args.update(hyperparameters)
    return model_args


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    model_loss_fn: Callable,
    classes: List,
    device: str = "cuda",
):
    num_classes = len(classes)
    total, total_correct, total_loss = 0, 0, 0
    predictions, labels = [], []
    with torch.no_grad():
        for fname, label, graphs in tqdm(dataloader):
            graphs, label = graphs.to(device), label.to(device)
            output = model(graphs)
            prediction = output.argmax(dim=1)
            correct_tensor = prediction == label
            prediction = torch.stack(
                [
                    torch.nn.functional.one_hot(p, num_classes)
                    for p in prediction.to(int)
                ]
            )
            predictions.append(prediction)
            label = torch.stack(
                [torch.nn.functional.one_hot(l, num_classes) for l in label.to(int)]
            )  # expand labels tensor to match output
            label = label.to(device, dtype=torch.float64)
            labels.append(label)

            if label.ndim == 0:
                total += 1
            else:
                total += label.shape[0]
            total_correct += correct_tensor.sum().item()
            total_loss += model_loss_fn(output, label)

        predictions = torch.concat(predictions)
        labels = torch.concat(labels)
        acc = total_correct / total
        precision, recall, f1, support = precision_recall_fscore_support(
            labels.to("cpu"),
            predictions.to("cpu"),
            labels=[i for i, clss in enumerate(classes)],
            zero_division=0.0,
        )
        stats_df = pd.DataFrame(
            {
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support%": support / total,
            }
        )
        stats_df.index = classes
        weighted_f1 = f1_score(
            labels.to("cpu", dtype=torch.float64),
            predictions.to("cpu", dtype=torch.float64),
            labels=classes,
            average="weighted",
            zero_division=0.0,
        )
        return {"stats": stats_df, "weighted_f1": weighted_f1, "loss": total_loss}


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    classes: List,
    hyperparameters: Dict[str, Any],
    out_dir: str,
    device: str = "cuda",
):
    num_classes = len(classes)
    model_loss_fn = torch.nn.BCEWithLogitsLoss()
    lr_name = "model_lr" if "model_lr" in hyperparameters else "lr"
    model_trainer = torch.optim.Adam(model.parameters(), lr=hyperparameters[lr_name])
    model_scheduler = ReduceLROnPlateau(
        model_trainer, "min", patience=0, cooldown=5, min_lr=hyperparameters["min_lr"]
    )
    peak_dev_f1 = (-1, 0)  # keep track of the best f1 + epoch at which it was reached
    for epoch in range(hyperparameters["num_epochs"]):
        epoch_model_loss = 0
        for i, (fnames, labels, graphs) in enumerate(tqdm(train_loader)):
            graphs = graphs.to(device)
            labels = torch.stack(
                [
                    torch.nn.functional.one_hot(label, num_classes)
                    for label in labels.to(int)
                ]
            )  # expand labels tensor to match output
            labels = labels.to(device, dtype=torch.float64)
            output = model(graphs)
            model_loss = model_loss_fn(output, labels)
            model_loss.backward()
            epoch_model_loss += model_loss.item()
            if ((i + 1) % hyperparameters["accum_iter"] == 0) or (
                i + 1 == len(train_loader)
            ):
                model_trainer.step()
                model_trainer.zero_grad()
        logging.info(f"Epoch {epoch + 1} model loss = {epoch_model_loss:.4}")
        tr_stats = evaluate(
            model, train_loader, model_loss_fn, classes=classes, device=device
        )
        logging.info(f"Training loss: {tr_stats['loss'].item():.4}")
        logging.info(f"Training stats: \n{tr_stats['stats']}")
        dev_stats = evaluate(
            model, dev_loader, model_loss_fn, classes=classes, device=device
        )
        logging.info(f"Dev loss: {dev_stats['loss'].item():.4}")
        logging.info(f"Dev stats: \n{dev_stats['stats']}")
        model_scheduler.step(dev_stats["loss"] / len(dev_loader))
        if dev_stats["stats"]["f1"][1:].mean() >= peak_dev_f1[0]:
            peak_dev_f1 = (dev_stats["stats"]["f1"][1:].mean(), epoch + 1)
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    logging.info("******** Training complete ********")
    logging.info(f"Peak dev minority f1 = {peak_dev_f1[0]} at epoch {peak_dev_f1[1]}")
    if test_loader is not None:
        if args.save_model:
            model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt")))

        test_stats = evaluate(
            model,
            test_loader,
            model_loss_fn,
            classes=classes,
            device=device,
            out_dir=out_dir,
        )
        logging.info(f"Test stats: \n{test_stats['stats']}")
    if args.delete_model:
        os.remove(os.path.join(out_dir, "model.pt"))


def main():
    device = get_device()
    # get hyperparameters
    if args.hyperparameters == "stdout":
        hyperparameters = json.loads(sys.stdin.read())
    elif isinstance(args.hyperparameters, str):
        with open(args.hyperparameters, "r") as f:
            hyperparameters = json.load(f)

    # set random seed for reporoducibility
    torch.manual_seed(hyperparameters["random_seed"])

    # set name for out files
    if (
        "SLURM_ARRAY_JOB_ID" in os.environ
    ):  # bundle together directories for runs in same slurm batch job
        task_id = os.path.join(
            os.environ["SLURM_ARRAY_JOB_ID"], os.environ["SLURM_ARRAY_TASK_ID"]
        )
    else:
        task_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(
        args.save_dir,
        hyperparameters["model_type"],
        "_".join(args.task_names).lower(),
        task_id,
    )
    logging_config(out_dir, "train", level=logging.DEBUG)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(json.dumps(hyperparameters))

    # get dataloaders
    homogenize = (
        True
        if hyperparameters["batch_size"] > 1
        or hyperparameters["model_type"] == "HGT"
        or (
            len(
                hyperparameters["motifs_to_use"] > 1
                and 0 in hyperparameters["motifs_to_use"]
            )
        )
        else False
    )
    vocab, labels, classes, train_loader, dev_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        labels_path=args.label_path,
        pretrained_dir=args.pretrained_dir,
        cache=args.cache,
        task_names=args.task_names,
        num_data=hyperparameters["num_data"],
        data_split=hyperparameters["data_split"],
        batch_size=hyperparameters["batch_size"],
        oversample=hyperparameters["oversample"],
        homogenize=homogenize,
        emb_dim=hyperparameters["emb_input_dim"],
        use_doctime=hyperparameters["use_doctime"],
        motifs_to_use=hyperparameters["motifs_to_use"],
        motifs_dir=args.motifs_dir,
    )

    vocab_size = len(vocab)

    model_args = get_model_args(
        vocab_size=vocab_size,
        labels=labels,
        num_classes=len(classes),
        hyperparameters=hyperparameters,
    )
    model = get_model(
        hyperparameters=hyperparameters,
        model_args=model_args,
    ).to(device)

    train_classifier(
        model,
        model_args,
        train_loader,
        dev_loader,
        test_loader,
        device=device,
        classes=classes,
        hyperparameters=hyperparameters,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
