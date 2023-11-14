import torch

from src.inputs.mnist_256x6 import solver_inputs
from src.solve import solve
from src.utils import seed_everything

seed_everything(0)
solve(solver_inputs, device=torch.device("cpu"))

print("Training finished!")
