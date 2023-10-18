import torch

from inputs.mnist_256x6 import solver_inputs
from modules.Solver import Solver
from solve import train

solver = Solver(solver_inputs)

train(solver)

print("Training finished!")
