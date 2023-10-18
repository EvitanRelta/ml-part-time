import torch

from inputs.mnist_256x6 import solver_inputs
from modules.Solver import Solver

solver = Solver(solver_inputs)

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
    loss, theta = solver.forward()

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
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {-loss.item()}")

print("Training finished!")
