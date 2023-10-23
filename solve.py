from typing import Literal, overload

import torch
from torch import Tensor

from modules.Solver import Solver
from preprocessing.solver_inputs import SolverInputs
from train import train


# fmt: off
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False, device: torch.device = torch.device('cpu')) -> tuple[Literal[True], list[Tensor], list[Tensor]]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False, device: torch.device = torch.device('cpu')) -> tuple[Literal[False], None, None]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True], device: torch.device = torch.device('cpu')) -> tuple[Literal[True], list[Tensor], list[Tensor], Solver]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True], device: torch.device = torch.device('cpu')) -> tuple[Literal[False], None, None, Solver]: ...
# fmt: on
def solve(
    solver_inputs: SolverInputs,
    return_solver: bool = False,
    device: torch.device = torch.device("cpu"),
) -> (
    tuple[bool, list[Tensor] | None, list[Tensor] | None]
    | tuple[bool, list[Tensor] | None, list[Tensor] | None, Solver]
):
    """
    Args:
        solver_inputs (SolverInputs): Dataclass containing all the inputs needed to start solving.

    Returns:
        `(is_falsified, new_lower_bounds, new_upper_bounds)`
    """
    solver = Solver(solver_inputs).to(device)

    new_L: list[Tensor] = []
    new_U: list[Tensor] = []
    for layer_index in range(len(solver.layers) - 1):  # Don't solve for last layer
        solver.reset_and_solve_for_layer(layer_index)
        if not train(solver):
            return (False, None, None, solver) if return_solver else (False, None, None)

        new_L_i, new_U_i = solver.get_updated_bounds(layer_index)
        new_L.append(new_L_i)
        new_U.append(new_U_i)

    # Add last initial bounds.
    new_L.append(solver.vars.L[-1])
    new_U.append(solver.vars.U[-1])

    return (True, new_L, new_U, solver) if return_solver else (True, new_L, new_U)
