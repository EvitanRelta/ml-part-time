import torch

from inputs.toy_example import solver_inputs
from solve import solve
from utils import seed_everything

seed_everything(0)
solve(solver_inputs, device=torch.device("cpu"))

print("Training finished!")
