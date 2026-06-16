"""Deterministic seeding across python / numpy / torch / cuda (SPEC §2.5)."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every RNG we depend on.

    ``deterministic_torch`` sets cuDNN to deterministic mode. We keep torch
    imports local so that non-training code paths (datagen, stats) need not pay
    the torch import cost.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep, but keep this import-safe
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
