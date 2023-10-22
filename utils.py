import random

import numpy as np
import onnx
import onnx2torch
import torch
from torch import Tensor, nn
from typing_extensions import List

from inputs.save_file_types import SolverInputsSavedDict


def seed_everything(seed: int) -> None:
    """Seeds `random`, `numpy`, `torch` with `seed` and makes computation deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_onnx_model(onnx_file_path: str) -> nn.Module:
    onnx_model = onnx.load(onnx_file_path)
    return onnx2torch.convert(onnx_model)


def convert_and_save_solver_inputs(
    lbounds: List[np.ndarray],
    ubounds: List[np.ndarray],
    Pall: List[np.ndarray],
    Phatall: List[np.ndarray],
    smallpall: List[np.ndarray],
    Hmatrix: np.ndarray,
    dvector: np.ndarray,
    save_filename: str = "dump.pth",
) -> None:
    """Converts raw solver inputs from numpy arrays to pytorch, then saves them
    to a Pytorch `.pth` file specified by the `save_filename` parameter.
    """
    # Check that the variable types are correct.
    list_error_msg = "`{var}` is not type `List[np.ndarray]`."
    array_error_msg = "`{var}` is not type `np.ndarray`."
    assert isinstance(lbounds, List) and all(
        isinstance(item, np.ndarray) for item in lbounds
    ), list_error_msg.format(var="lbounds")
    assert isinstance(ubounds, List) and all(
        isinstance(item, np.ndarray) for item in ubounds
    ), list_error_msg.format(var="ubounds")
    assert isinstance(Pall, List) and all(
        isinstance(item, np.ndarray) for item in Pall
    ), list_error_msg.format(var="Pall")
    assert isinstance(Phatall, List) and all(
        isinstance(item, np.ndarray) for item in Phatall
    ), list_error_msg.format(var="Phatall")
    assert isinstance(smallpall, np.ndarray), array_error_msg.format(var="smallpall")
    assert isinstance(Hmatrix, np.ndarray), array_error_msg.format(var="Hmatrix")
    assert isinstance(dvector, np.ndarray), array_error_msg.format(var="dvector")

    # Convert numpy arrays to PyTorch tensors and then into the formats used in the solver.
    L: List[Tensor] = [torch.tensor(x).float().squeeze() for x in lbounds]
    U: List[Tensor] = [torch.tensor(x).float().squeeze() for x in ubounds]
    H: Tensor = torch.tensor(Hmatrix).float().squeeze()
    d: Tensor = torch.tensor(dvector).float().squeeze()
    P: List[Tensor] = [torch.tensor(x).float().squeeze() for x in Pall]
    P_hat: List[Tensor] = [torch.tensor(x).float().squeeze() for x in Phatall]
    p: List[Tensor] = [torch.tensor(x).float().squeeze() for x in smallpall]

    # Save to file.
    saved_dict: SolverInputsSavedDict = {
        "L": L,
        "U": U,
        "H": H,
        "d": d,
        "P": P,
        "P_hat": P_hat,
        "p": p,
    }
    torch.save(saved_dict, save_filename)
