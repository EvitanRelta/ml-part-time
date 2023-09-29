import torch

from inputs.toy_example import H, L, P, P_hat, U, _alpha, _gamma, _pi, d, model, p
from inputs_dataclasses import SolverInputs
from Solver import Solver

inputs = SolverInputs(
    model=model,
    L=L,
    U=U,
    H=H,
    d=d,
    P=P,
    P_hat=P_hat,
    p=p,
    initial_gamma=_gamma,
    initial_pi=_pi,
    initial_alpha=_alpha,
)
solver = Solver(inputs)

# Set minimisation of x4 as target.
solver.set_target(1, 1, is_min=True)

# Define the optimizer
learning_rate = 0.01
optimizer = torch.optim.Adam(solver.parameters(), lr=learning_rate)

# Number of epochs (complete passes over the data)
n_epochs = 10000

# Threshold for the minimum change in loss
threshold = 1e-5

# Initialize variable for previous loss
prev_loss = float("inf")

# Gradient Descent
for epoch in range(n_epochs):
    # Forward pass: Compute loss
    loss = solver.forward()

    # Check if the change in loss is less than the threshold, if so, stop training
    if abs(prev_loss - loss.item()) < threshold:
        print(f"Training stopped at epoch {epoch}, Loss: {-loss.item()}")
        break

    # Update previous loss
    prev_loss = loss.item()

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Compute the gradients
    optimizer.step()  # Update the parameters

    solver.clamp_parameters()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {-loss.item()}")

print("Training finished!")
