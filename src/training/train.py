from typing import List

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.autonotebook import tqdm

from ..modules.Solver import Solver
from .EarlyStopHandler import EarlyStopHandler
from .TrainingConfig import TrainingConfig


def train(solver: Solver, config: TrainingConfig = TrainingConfig()) -> bool:
    """Train `solver` until convergence or until the problem is falsified, and
    return whether the problem was falsified.

    - Returns `False` if `solver` was trained to convergence without problems.
    - Returns `True` if training was stopped prematurely due to being falsified.

    Args:
        solver (Solver): The `Solver` model to train.
        config (TrainingConfig, optional): Configuration to use during training. \
            Defaults to TrainingConfig().

    Returns:
        bool: Whether the problem was falsified. `False` if `solver` was trained to \
            convergence, `True` if training was stopped prematurely due to being falsified.
    """
    optimizer = Adam(solver.parameters(), config.max_lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        threshold=config.reduce_lr_threshold,
        min_lr=config.min_lr,
    )
    early_stop_handler = EarlyStopHandler(config.min_lr)

    theta_list: List[Tensor] = []

    epoch = 1
    pbar = tqdm(
        desc="Training",
        total=None,
        unit=" epoch",
        initial=epoch,
        disable=config.disable_progress_bar,
    )
    while True:
        max_objective, theta = solver.forward()
        if not config.disable_adv_check:
            # Accumulate thetas for later concrete-input adversarial checking.
            theta_list.append(theta)

        loss = -max_objective.sum()
        loss_float = loss.item()

        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss_float)

        # Clamp learnable parameters to their respective value ranges.
        solver.clamp_parameters()

        if not config.disable_adv_check and epoch % config.num_epoch_adv_check == 0:
            # Check if accumulated thetas fails adversarial check.
            # If it fails, stop prematurely. If it passes, purge the
            # accumulated thetas to free up memory.
            if is_falsified_by_concrete_inputs(solver, theta_list):
                return True
            theta_list = []

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"Loss": loss_float, "LR": current_lr})
        pbar.update()

        if early_stop_handler.is_early_stopped(current_lr):
            pbar.set_description(f"Training stopped at epoch {epoch}, Loss: {loss_float}")
            pbar.close()
            break

        epoch += 1

    if (
        not config.disable_adv_check
        and len(theta_list) > 0
        and is_falsified_by_concrete_inputs(solver, theta_list)
    ):
        return True

    return False


def is_falsified_by_concrete_inputs(solver: Solver, theta_list: List[Tensor]) -> bool:
    """Whether concrete inputs generated from `theta_list` falsifies the problem
    via the adversarial-check model (ie. training should be stopped).
    """
    thetas = torch.cat(theta_list, dim=0)
    L_0: Tensor = solver.layers.solvable_layers[0].L.detach()
    U_0: Tensor = solver.layers.solvable_layers[0].U.detach()
    concrete_inputs: Tensor = torch.where(thetas >= 0, L_0, U_0)
    return solver.adv_check_model.forward(concrete_inputs)
