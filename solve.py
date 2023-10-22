from typing import Literal, overload

import torch
from torch import Tensor

from modules.Solver import Solver
from preprocessing.solver_inputs import SolverInputs
from train import train


# fmt: off
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False) -> tuple[Literal[True], list[Tensor], list[Tensor]]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False) -> tuple[Literal[False], None, None]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True]) -> tuple[Literal[True], list[Tensor], list[Tensor], Solver]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True]) -> tuple[Literal[False], None, None, Solver]: ...
# fmt: on
def solve(
    solver_inputs: SolverInputs,
    return_solver: bool = False,
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
    solver = Solver(solver_inputs)

    new_L: list[Tensor] = []
    new_U: list[Tensor] = []
    for layer_index in range(len(solver.layers) - 1):  # Don't solve for last layer
        solver.reset_and_solve_for_layer(layer_index)
        thetas = train(solver)

        new_L_i, new_U_i = solver.get_updated_bounds(layer_index)
        new_L.append(new_L_i)
        new_U.append(new_U_i)

        L_0: Tensor = solver.vars.layer_vars[0].L_i
        U_0: Tensor = solver.vars.layer_vars[0].U_i
        concrete_inputs: Tensor = torch.where(thetas >= 0, L_0, U_0)
        unique_concrete_inputs = torch.unique(concrete_inputs, dim=0)
        assert isinstance(unique_concrete_inputs, Tensor)

        if solver.adv_check_model.forward(unique_concrete_inputs):
            return (False, None, None, solver) if return_solver else (False, None, None)

    # Add last initial bounds.
    new_L.append(solver.vars.inputs.L[-1])
    new_U.append(solver.vars.inputs.U[-1])

    return (True, new_L, new_U, solver) if return_solver else (True, new_L, new_U)
