import torch

from src.compare_against_gurobi import compare_against_gurobi
from src.inputs.mnist_256x6 import gurobi_results, solver_inputs
from src.solve import solve
from src.utils import seed_everything

seed_everything(0)
is_falsified, new_L, new_U, solver = solve(
    solver_inputs,
    device=torch.device("cuda"),
    return_solver=True,
)

unstable_masks = solver.vars.unstable_masks

compare_against_gurobi(
    new_L=new_L,
    new_U=new_U,
    unstable_masks=unstable_masks,
    initial_L=solver_inputs.L,
    initial_U=solver_inputs.U,
    gurobi_results=gurobi_results,
    cutoff_threshold=1e-5,
)
