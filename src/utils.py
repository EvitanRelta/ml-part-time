import os
import random
from typing import Callable

import numpy as np
import onnx
import onnx2torch
import torch
from torch import nn


def set_abs_path_to(current_dir: str) -> Callable[[str], str]:
    """Higher-order-function for getting absolute paths relative to `current_dir`.

    Examples:
        ```python
        CURRENT_DIR = os.path.dirname(__file__)
        get_abs_path = set_abs_path_to(CURRENT_DIR)

        # Gets absolute path of `file.ext` that's in the
        # same dir as the current file.
        get_abs_path("file.ext")
        ```
    """
    return lambda path: os.path.join(current_dir, path)


def seed_everything(seed: int) -> None:
    """Seeds `random`, `numpy`, `torch` with `seed` and makes computation deterministic."""
    random.seed(seed)
    np.random.seed(seed)  # type: ignore
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_onnx_model(onnx_file_path: str) -> nn.Module:
    onnx_model = onnx.load(onnx_file_path)
    return onnx2torch.convert(onnx_model)
