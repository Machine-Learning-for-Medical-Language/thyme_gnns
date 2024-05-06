import os
import logging
import torch

__all__ = ["logging_config", "get_device"]


def get_device():
    # if torch.has_cuda:
    if torch.backends.cuda.is_built():
        return torch.device("cuda")
    # elif torch.has_mps:
    if torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def logging_config(
    out_dir=None,
    name=None,
    level=logging.INFO,
    console_level=logging.INFO,
):
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(out_dir, name + ".log")
    logging.root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    # console logging
    logconsole = logging.StreamHandler()
    logconsole.setLevel(console_level)
    logconsole.setFormatter(formatter)
    logging.root.addHandler(logconsole)
    return out_dir
