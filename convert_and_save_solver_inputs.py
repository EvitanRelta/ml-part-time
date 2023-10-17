import numpy as np
import torch
from torch import Tensor
from typing_extensions import List


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
    torch.save(
        {
            "L": L,
            "U": U,
            "H": H,
            "d": d,
            "P": P,
            "P_hat": P_hat,
            "p": p,
        },
        save_filename,
    )
