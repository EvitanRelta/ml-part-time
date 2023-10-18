from torch.optim import Adam
from tqdm.autonotebook import tqdm

from modules.Solver import Solver


def train(solver: Solver, lr: float = 1e-2, stop_threshold: float = 1e-5, max_epoches: int = 5000):
    optimizer = Adam(solver.parameters(), lr)
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

        solver.clamp_parameters()

        # Set the description for tqdm
        pbar.set_description(f"Epoch {epoch}")
        pbar.set_postfix({"Loss": -loss.item()})  # Use set_postfix to display the loss

    return
