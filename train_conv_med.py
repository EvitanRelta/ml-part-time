import torch

from .inputs.conv_med import solver_inputs
from .solve import solve
from .utils import seed_everything

seed_everything(0)
solve(solver_inputs, device=torch.device("cpu"))

print("Training finished!")
