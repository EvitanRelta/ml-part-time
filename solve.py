from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.autonotebook import tqdm

from modules.Solver import Solver


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

    # Create tqdm object
    pbar = tqdm(range(max_epoches), desc="Training", unit="epoch")

    for epoch in pbar:
        loss, theta = solver.forward()

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

    return
