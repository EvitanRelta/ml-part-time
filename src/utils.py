import os
import random
from typing import Callable, List, Literal, Tuple, Union, overload

import numpy as np
import onnx
import onnx2torch
import torch
from torch.fx.graph_module import GraphModule


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


# fmt: off
@overload
def load_onnx_model(onnx_file_path: str, return_input_shape: Literal[False] = False) -> GraphModule: ...
@overload
def load_onnx_model(onnx_file_path: str, return_input_shape: Literal[True]) -> Tuple[GraphModule, Tuple[int, ...]]: ...
# fmt: on
def load_onnx_model(onnx_file_path: str, return_input_shape: bool = False) -> Union[GraphModule, Tuple[GraphModule, Tuple[int, ...]]]:  # fmt: skip
    """Loads an ONNX model from a path to an `.onnx` file, and convert it to a PyTorch module.

    Can also optionally return the ONNX model's input shape via `return_input_shape=True`.

    Args:
        onnx_file_path (str): Path to `.onnx` ONNX model save-file.
        return_input_shape (bool, optional): Whether to also return the ONNX \
            model's input shape. Defaults to False.

    Returns:
        Union[GraphModule, Tuple[GraphModule, Tuple[int, ...]]]: The loaded \
            ONNX model converted to a PyTorch module, and optionally, the ONNX \
            model's input shape when `return_input_shape=True`.
    """
    onnx_model = onnx.load(onnx_file_path)
    return (
        (onnx2torch.convert(onnx_model), get_onnx_input_shape(onnx_model))
        if return_input_shape
        else onnx2torch.convert(onnx_model)
    )


def get_onnx_input_shape(onnx_model: onnx.ModelProto) -> Tuple[int, ...]:
    """Gets the ONNX model's input shape.

    Note: Assumes that there's exactly 1 input.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model to get the input shape from.

    Returns:
        Tuple[int, ...]: The ONNX model's input shape.
    """
    assert len(onnx_model.graph.input) == 1
    output: List[int] = []
    onnx_dims = iter(onnx_model.graph.input[0].type.tensor_type.shape.dim)

    # Handle first dim, which sometimes might not have a `.dim_value`.
    dim_1 = next(onnx_dims)
    if dim_1.HasField("dim_value"):
        output.append(dim_1.dim_value)
    else:
        assert dim_1.HasField("dim_param"), "Unknown 1st dim in ONNX model's input."
        output.append(1)  # Assume this is batch_size, and set batch_size=1.

    for x in onnx_dims:
        assert x.HasField("dim_value"), "Unknown dim in ONNX model's input."
        output.append(x.dim_value)
    return tuple(output)
