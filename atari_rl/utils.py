"""Small shared helpers: reproducible seeding and a minimal CSV logger."""

from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic_torch=True`` also forces deterministic cuDNN (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def add_bool_flag(parser, name: str, default: bool, help: str = "") -> None:
    """Add a ``--name`` / ``--no-name`` boolean flag pair.

    Like ``argparse.BooleanOptionalAction``, but works on Python 3.8.
    """
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", help=help)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


class CSVLogger:
    """Append (step, value, ...) rows to a CSV file, writing a header once."""

    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(fieldnames)

    def log(self, *values) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(values)
