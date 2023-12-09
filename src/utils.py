import os
import random
from typing import Callable, Union

import numpy as np
import onnx
import onnx2torch
import torch
from torch import Tensor, nn
from typing_extensions import List

from .inputs.save_file_types import SolverInputsSavedDict
from .preprocessing.solver_inputs import SolverInputs


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_onnx_model(onnx_file_path: str) -> nn.Module:
    onnx_model = onnx.load(onnx_file_path)
    return onnx2torch.convert(onnx_model)


def convert_and_save_solver_inputs(
    lbounds: Union[List[np.ndarray], List[Tensor]],
    ubounds: Union[List[np.ndarray], List[Tensor]],
    Pall: Union[List[np.ndarray], List[Tensor]],
    Phatall: Union[List[np.ndarray], List[Tensor]],
    smallpall: Union[List[np.ndarray], List[Tensor]],
    Hmatrix: Union[np.ndarray, Tensor],
    dvector: Union[np.ndarray, Tensor],
    ground_truth_neuron_index: int,
    save_filename: str = "dump.pth",
) -> None:
    """Converts raw solver inputs from numpy-arrays / pytorch-tensors to the
    expected input type/shape, then saves them to a Pytorch `.pth` file
    specified by the `save_filename` parameter.
    """
    # Check that the variable types are correct.
    list_error_msg = "`{var}` is not type `List[np.ndarray]`."
    array_error_msg = "`{var}` is not type `np.ndarray`."
    assert isinstance(lbounds, list) and all(
        isinstance(item, np.ndarray) or isinstance(item, Tensor) for item in lbounds
    ), list_error_msg.format(var="lbounds")
    assert isinstance(ubounds, list) and all(
        isinstance(item, np.ndarray) or isinstance(item, Tensor) for item in ubounds
    ), list_error_msg.format(var="ubounds")
    assert isinstance(Pall, list) and all(
        isinstance(item, np.ndarray) or isinstance(item, Tensor) for item in Pall
    ), list_error_msg.format(var="Pall")
    assert isinstance(Phatall, list) and all(
        isinstance(item, np.ndarray) or isinstance(item, Tensor) for item in Phatall
    ), list_error_msg.format(var="Phatall")
    assert isinstance(smallpall, list) and all(
        isinstance(item, np.ndarray) or isinstance(item, Tensor) for item in smallpall
    ), list_error_msg.format(var="smallpall")
    assert isinstance(Hmatrix, np.ndarray) or isinstance(Hmatrix, Tensor), array_error_msg.format(
        var="Hmatrix"
    )
    assert isinstance(dvector, np.ndarray) or isinstance(dvector, Tensor), array_error_msg.format(
        var="dvector"
    )
    assert isinstance(
        ground_truth_neuron_index, int
    ), "`ground_truth_neuron_index` is not type `int`."

    # Convert numpy arrays to PyTorch tensors and then into the formats used in the solver.
    L_list: List[Tensor] = [torch.atleast_1d(torch.tensor(x).float().squeeze()) for x in lbounds]
    U_list: List[Tensor] = [torch.atleast_1d(torch.tensor(x).float().squeeze()) for x in ubounds]
    H: Tensor = torch.atleast_2d(torch.tensor(Hmatrix).float().squeeze())
    d: Tensor = torch.atleast_1d(torch.tensor(dvector).float().squeeze())
    P_list: List[Tensor] = [torch.atleast_2d(torch.tensor(x).float().squeeze()) for x in Pall]
    P_hat_list: List[Tensor] = [
        torch.atleast_2d(torch.tensor(x).float().squeeze()) for x in Phatall
    ]
    p_list: List[Tensor] = [torch.atleast_1d(torch.tensor(x).float().squeeze()) for x in smallpall]

    # Save to file.
    saved_dict: SolverInputsSavedDict = {
        "L_list": L_list,
        "U_list": U_list,
        "H": H,
        "d": d,
        "P_list": P_list,
        "P_hat_list": P_hat_list,
        "p_list": p_list,
        "ground_truth_neuron_index": ground_truth_neuron_index,
    }
    torch.save(saved_dict, save_filename)


def ensure_tensor(array_or_tensor: Union[np.ndarray, Tensor]) -> Tensor:
    """Converts `array_or_tensor` to a Pytorch Tensor if necessary."""
    return (
        torch.from_numpy(array_or_tensor)
        if isinstance(array_or_tensor, np.ndarray)
        else array_or_tensor
    )


def wrap_solver_inputs(
    lbounds: Union[List[np.ndarray], List[Tensor]],
    ubounds: Union[List[np.ndarray], List[Tensor]],
    Pall: Union[List[np.ndarray], List[Tensor]],
    Phatall: Union[List[np.ndarray], List[Tensor]],
    smallpall: Union[List[np.ndarray], List[Tensor]],
    Hmatrix: Union[np.ndarray, Tensor],
    dvector: Union[np.ndarray, Tensor],
    ground_truth_neuron_index: int,
    model: nn.Module,
) -> SolverInputs:
    L_list: List[Tensor] = [torch.atleast_1d(ensure_tensor(x).float().squeeze()) for x in lbounds]
    U_list: List[Tensor] = [torch.atleast_1d(ensure_tensor(x).float().squeeze()) for x in ubounds]
    H: Tensor = torch.atleast_2d(ensure_tensor(Hmatrix).float().squeeze())
    d: Tensor = torch.atleast_1d(ensure_tensor(dvector).float().squeeze())
    P_list: List[Tensor] = [torch.atleast_2d(ensure_tensor(x).float().squeeze()) for x in Pall]
    P_hat_list: List[Tensor] = [
        torch.atleast_2d(ensure_tensor(x).float().squeeze()) for x in Phatall
    ]
    p_list: List[Tensor] = [torch.atleast_1d(ensure_tensor(x).float().squeeze()) for x in smallpall]
    return SolverInputs(
        model=model,
        ground_truth_neuron_index=ground_truth_neuron_index,
        L_list=L_list,
        U_list=U_list,
        H=H,
        d=d,
        P_list=P_list,
        P_hat_list=P_hat_list,
        p_list=p_list,
    )
