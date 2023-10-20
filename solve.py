from typing import Literal, overload

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.autonotebook import tqdm

from modules.Solver import Solver
from preprocessing.solver_inputs import SolverInputs


# fmt: off
@overload
def solve(solver_inputs: SolverInputs) -> tuple[Literal[True], list[Tensor], list[Tensor]]: ...
@overload
def solve(solver_inputs: SolverInputs) -> tuple[Literal[False], None, None]: ...
# fmt: on
def solve(solver_inputs: SolverInputs) -> tuple[bool, list[Tensor] | None, list[Tensor] | None]:
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
            return False, None, None
    return True, new_L, new_U


def train(solver: Solver, lr: float = 1, stop_threshold: float = 1e-4, max_epoches: int = 10000):
    optimizer = Adam(solver.parameters(), lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
    )
    prev_loss = float("inf")
    theta_list: list[Tensor] = []

    # Create tqdm object
    pbar = tqdm(range(max_epoches), desc="Training", unit="epoch")

    for epoch in pbar:
        max_objective, theta = solver.forward()
        loss = -max_objective.sum()
        theta_list.append(theta)

        # Check if the change in loss is less than the threshold, if so, stop training
        if abs(prev_loss - loss.item()) < stop_threshold:
            pbar.set_description(f"Training stopped at epoch {epoch}, Loss: {-loss.item()}")
            pbar.close()  # Close the tqdm loop when training stops
            break

        prev_loss = loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        solver.clamp_parameters()

        # Set the description for tqdm
        pbar.set_description(f"Epoch {epoch}")
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"Loss": -loss.item(), "LR": current_lr})

    return torch.cat(theta_list, dim=0)
