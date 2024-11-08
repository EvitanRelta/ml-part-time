import dataclasses
from typing import List, Literal, Tuple, Union, overload

import torch
from numpy import ndarray
from torch import Tensor

from .modules.Solver import Solver
from .preprocessing.solver_inputs import SolverInputs
from .training.train import train
from .training.TrainingConfig import TrainingConfig


# fmt: off
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False, device: torch.device = torch.device('cpu'), training_config: TrainingConfig = TrainingConfig()) -> Tuple[Literal[True], None, None]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[False] = False, device: torch.device = torch.device('cpu'), training_config: TrainingConfig = TrainingConfig()) -> Tuple[Literal[False], List[ndarray], List[ndarray]]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True], device: torch.device = torch.device('cpu'), training_config: TrainingConfig = TrainingConfig()) -> Tuple[Literal[True], None, None, Solver]: ...
@overload
def solve(solver_inputs: SolverInputs, return_solver: Literal[True], device: torch.device = torch.device('cpu'), training_config: TrainingConfig = TrainingConfig()) -> Tuple[Literal[False], List[ndarray], List[ndarray], Solver]: ...
# fmt: on
def solve(
    solver_inputs: SolverInputs,
    return_solver: bool = False,
    device: torch.device = torch.device("cpu"),
    training_config: TrainingConfig = TrainingConfig(),
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
        training_config (TrainingConfig, optional): Configuration to use during training. \
            Defaults to TrainingConfig().

    Returns:
        `(is_falsified, new_lower_bounds, new_upper_bounds)` and optionally, the `Solver` instance \
            as the last element if `return_solver == True`.
    """
    solver = Solver(solver_inputs).to(device)

    new_L_list: List[Tensor] = []
    new_U_list: List[Tensor] = []
    for layer_index in range(len(solver.layers.solvable_layers)):
        for _ in range(3):  # try training 3 times, reducing starting LR up to 0.001
            solver.reset_and_solve_for_layer(layer_index)
            is_train_success, is_falsified = train(solver, training_config)
            if is_train_success:
                break
            # If training failed due to too high LR, reduce LR and try again.
            print("Training failed. Reducing LR and trying again...")
            training_config = dataclasses.replace(training_config)  # copy config
            training_config.max_lr = training_config.max_lr * 0.1  # reduce to 10% of previous LR

        if not is_train_success:
            raise Exception("Failed to train.")

        if is_falsified:
            return (True, None, None, solver) if return_solver else (True, None, None)

        new_L, new_U = solver.get_updated_bounds(layer_index)
        new_L_list.append(new_L)
        new_U_list.append(new_U)

    # Convert tensors to numpy arrays.
    numpy_L_list: List[ndarray] = [x.cpu().numpy() for x in new_L_list]
    numpy_U_list: List[ndarray] = [x.cpu().numpy() for x in new_U_list]

    return (
        (False, numpy_L_list, numpy_U_list, solver)
        if return_solver
        else (False, numpy_L_list, numpy_U_list)
    )
