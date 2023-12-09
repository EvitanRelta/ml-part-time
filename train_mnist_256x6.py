import torch

from src.compare_against_gurobi import compare_against_gurobi
from src.inputs.mnist_256x6 import gurobi_results, solver_inputs
from src.solve import solve
from src.utils import seed_everything

seed_everything(0)
is_falsified, new_L_list, new_U_list, solver = solve(
    solver_inputs,
    device=torch.device("cuda"),
    return_solver=True,
)

if is_falsified:
    print("Verification problem is falsified.")
    exit(0)

unstable_masks = solver.vars.unstable_masks

compare_against_gurobi(
    new_L_list=[torch.from_numpy(x) for x in new_L_list],
    new_U_list=[torch.from_numpy(x) for x in new_U_list],
    unstable_masks=unstable_masks,
    initial_L_list=solver_inputs.L_list,
    initial_U_list=solver_inputs.U_list,
    gurobi_results=gurobi_results,
    cutoff_threshold=1e-5,
)
