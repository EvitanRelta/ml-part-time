from typing import List, Literal, Tuple, Union, overload

import torch
from numpy import ndarray
from torch import Tensor

from .modules.Solver import Solver
from .preprocessing.solver_inputs import SolverInputs
from .train import train


# fmt: off
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False, device: torch.device = torch.device('cpu'), num_epoch_adv_check: int = 10, run_adv_check: bool = True) -> Tuple[Literal[True], List[ndarray], List[ndarray]]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False, device: torch.device = torch.device('cpu'), num_epoch_adv_check: int = 10, run_adv_check: bool = True) -> Tuple[Literal[False], None, None]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True], device: torch.device = torch.device('cpu'), num_epoch_adv_check: int = 10, run_adv_check: bool = True) -> Tuple[Literal[True], List[ndarray], List[ndarray], Solver]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True], device: torch.device = torch.device('cpu'), num_epoch_adv_check: int = 10, run_adv_check: bool = True) -> Tuple[Literal[False], None, None, Solver]: ...
# fmt: on
def solve(
    solver_inputs: SolverInputs,
    return_solver: bool = False,
    device: torch.device = torch.device("cpu"),
    num_epoch_adv_check: int = 10,
    run_adv_check: bool = True,
) -> Union[
    Tuple[bool, Union[List[ndarray], None], Union[List[ndarray], None]],
    Tuple[bool, Union[List[ndarray], None], Union[List[ndarray], None], Solver],
]:
    """
    Args:
        solver_inputs (SolverInputs): Dataclass containing all the inputs needed to start solving.
        return_solver (bool, optional): Whether to also return the `Solver` instance. \
            Defaults to False.
        device (torch.device, optional): Device to compute on. Defaults to torch.device("cpu").
        num_epoch_adv_check (int, optional): Perform adversarial check every `num_epoch_adv_check`\
            epochs. Defaults to 10.
        run_adv_check (bool, optional): Whether to run the adversarial check. Defaults to True.

    Returns:
        `(is_falsified, new_lower_bounds, new_upper_bounds)` and optionally, the `Solver` instance \
            as the last element if `return_solver == True`.

    """
    solver = Solver(solver_inputs).to(device)

    new_L_list: List[Tensor] = []
    new_U_list: List[Tensor] = []
    for layer_index in range(len(solver.layers) - 1):  # Don't solve for last layer
        solver.reset_and_solve_for_layer(layer_index)
        is_falsified = train(
            solver,
            num_epoch_adv_check=num_epoch_adv_check,
            run_adv_check=run_adv_check,
        )
        if is_falsified:
            return (True, None, None, solver) if return_solver else (True, None, None)

        new_L, new_U = solver.get_updated_bounds(layer_index)
        new_L_list.append(new_L)
        new_U_list.append(new_U)

    # Add last initial bounds.
    new_L_list.append(solver.vars.L_list[-1])
    new_U_list.append(solver.vars.U_list[-1])

    # Convert tensors to numpy arrays.
    numpy_L_list: List[ndarray] = [x.cpu().numpy() for x in new_L_list]
    numpy_U_list: List[ndarray] = [x.cpu().numpy() for x in new_U_list]

    return (
        (False, numpy_L_list, numpy_U_list, solver)
        if return_solver
        else (False, numpy_L_list, numpy_U_list)
    )
