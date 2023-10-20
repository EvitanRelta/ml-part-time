import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.autonotebook import tqdm

from modules.Solver import Solver


def train(solver: Solver, lr: float = 1, stop_threshold: float = 1e-4):
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

    epoch = 1
    pbar = tqdm(desc="Training", total=None, unit=" epoch", initial=epoch)
    while True:
        max_objective, theta = solver.forward()
        loss = -max_objective.sum()
        loss_float: float = loss.item()
        theta_list.append(theta)

        # Check if the change in loss is less than the threshold, if so, stop training
        if abs(prev_loss - loss_float) < stop_threshold:
            pbar.set_description(f"Training stopped at epoch {epoch}, Loss: {loss_float}")
            pbar.close()  # Close the tqdm loop when training stops
            break

        prev_loss = loss_float

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
