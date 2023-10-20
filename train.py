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
    max_lr: float = 1,
    min_lr: float = 1e-5,
    stop_patience: int = 10,
    stop_threshold: float = 1e-4,
) -> Tensor:
    """Train `solver` until convergence.

    Args:
        solver (Solver): The `Solver` model to train.
        max_lr (float, optional): Max learning-rate. Defaults to 1.
        min_lr (float, optional): Min learning-rate to decay until. Defaults to 1e-5.
        stop_patience (int, optional): Num. of epochs with no improvement, after which training \
            should be stopped. Defaults to 10.
        stop_threshold (float, optional): Threshold to determine whether there's "no improvement" \
            for early-stopping. No improvement is when `current_loss >= best_loss * (1 - threshold)`. \
            Defaults to 1e-4.

    Returns:
        Tensor: Accumulated batch of thetas, to be used for concrete-input adversarial checking. \
            Shape: `(num_solves * num_epoches, num_input_neurons)`.
    """
    optimizer = Adam(solver.parameters(), max_lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=min_lr,
    )
    early_stop_handler = EarlyStopHandler(stop_patience, stop_threshold)

    theta_list: list[Tensor] = []

    epoch = 1
    pbar = tqdm(desc="Training", total=None, unit=" epoch", initial=epoch)
    while True:
        max_objective, theta = solver.forward()
        loss = -max_objective.sum()
        loss_float: float = loss.item()
        theta_list.append(theta)

        # Check if the change in loss is less than the threshold, if so, stop training
        if early_stop_handler.is_early_stopped(loss_float):
            pbar.set_description(f"Training stopped at epoch {epoch}, Loss: {loss_float}")
            pbar.close()  # Close the tqdm loop when training stops
            break

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss_float)

        solver.clamp_parameters()

        # Set the description for tqdm
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"Loss": loss_float, "LR": current_lr})
        pbar.update()
        epoch += 1

    return torch.cat(theta_list, dim=0)
