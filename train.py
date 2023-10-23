import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.autonotebook import tqdm

from modules.Solver import Solver


class EarlyStopHandler:
    """Handler to determine whether to early-stop the training.

    Based on Pytorch's `ReduceLROnPlateau` scheduler's patience /
    relative-threshold.
    """

    def __init__(self, patience: int, threshold: float) -> None:
        """
        Args:
            patience (int): Num. of epochs with no improvement, after which training should be stopped.
            threshold (float): Threshold to determine whether there's "no improvement". \
                No improvement is when `current_loss >= best_loss * (1 - threshold)`.
        """
        self.patience = patience
        self.threshold = threshold
        self._num_no_improvements: int = 0
        self._best_loss: float = float("inf")

    def is_early_stopped(self, current_loss: float) -> bool:
        """Returns whether to stop the training early."""
        has_no_improvement = current_loss >= self._best_loss * (1 - self.threshold)
        if has_no_improvement:
            self._num_no_improvements += 1
        else:
            self._best_loss = current_loss
            self._num_no_improvements = 0

        return self._num_no_improvements >= self.patience


def train(
    solver: Solver,
    num_epoch_adv_check: int = 10,
    max_lr: float = 2,
    min_lr: float = 1e-6,
    stop_patience: int = 10,
    stop_threshold: float = 1e-3,
) -> bool:
    """Train `solver` until convergence.

    - Returns `True` if `solver` was trained to convergence without problems.
    - Returns `False` if training was stopped prematurely because it failed the
      adversarial check (ie. `is_falsified = False`).

    Args:
        solver (Solver): The `Solver` model to train.
        num_epoch_adv_check (int, optional): Perform adversarial check every `num_epoch_adv_check`\
            epochs.
        max_lr (float, optional): Max learning-rate. Defaults to 1.
        min_lr (float, optional): Min learning-rate to decay until. Defaults to 1e-5.
        stop_patience (int, optional): Num. of epochs with no improvement, after which training \
            should be stopped. Defaults to 10.
        stop_threshold (float, optional): Threshold to determine whether there's "no improvement" \
            for early-stopping. No improvement is when `current_loss >= best_loss * (1 - threshold)`. \
            Defaults to 1e-4.

    Returns:
        bool: `True` if `solver` is trained to convergence, `False` if training was stopped \
            prematurely from failed adversarial check.
    """
    optimizer = Adam(solver.parameters(), max_lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=3,
        threshold=0.001,
        min_lr=min_lr,
    )
    early_stop_handler = EarlyStopHandler(stop_patience, stop_threshold)

    theta_list: list[Tensor] = []

    epoch = 1
    pbar = tqdm(desc="Training", total=None, unit=" epoch", initial=epoch)
    while True:
        max_objective, theta = solver.forward()
        theta_list.append(theta)  # Accumulate thetas for later concrete-input adversarial checking.

        loss = -max_objective.sum()
        loss_float = loss.item()

        if early_stop_handler.is_early_stopped(loss_float):
            pbar.set_description(f"Training stopped at epoch {epoch}, Loss: {loss_float}")
            pbar.close()
            print()
            break

        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss_float)

        # Clamp learnable parameters to their respective value ranges.
        solver.clamp_parameters()

        if epoch % num_epoch_adv_check == 0:
            # Check if accumulated thetas fails adversarial check.
            # If it fails, stop prematurely. If it passes, purge the
            # accumulated thetas to free up memory.
            if fails_adv_check(solver, theta_list):
                return False
            theta_list = []

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"Loss": loss_float, "LR": current_lr})
        pbar.update()
        epoch += 1

    if len(theta_list) > 0 and fails_adv_check(solver, theta_list):
        return False

    return True


def fails_adv_check(solver: Solver, theta_list: list[Tensor]) -> bool:
    """Whether concrete inputs generated from `theta_list` fails the adversarial
    check (ie. training should be stopped).
    """
    thetas = torch.cat(theta_list, dim=0)
    L_0: Tensor = solver.vars.layer_vars[0].L_i.detach()
    U_0: Tensor = solver.vars.layer_vars[0].U_i.detach()
    concrete_inputs: Tensor = torch.where(thetas >= 0, L_0, U_0)
    return solver.adv_check_model.forward(concrete_inputs)
