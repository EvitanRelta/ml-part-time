import torch

from inputs.toy_example import solver_inputs
from modules.Solver import Solver
from solve import train

solver = Solver(solver_inputs)

train(solver)

print("Training finished!")
