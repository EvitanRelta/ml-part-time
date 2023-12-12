import os
import sys

import torch

from src.compare_against_gurobi import compare_against_gurobi
from src.inputs.mnist_256x6 import gurobi_results, solver_inputs
from src.solve import solve
from src.training.TrainingConfig import TrainingConfig
from src.utils import seed_everything, set_abs_path_to

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
CONFIG_FILE_PATH = get_abs_path("default_training_config.yaml")

seed_everything(0)

is_falsified, new_L_list, new_U_list, solver = solve(
    solver_inputs,
    device=torch.device("cuda"),
    return_solver=True,
    training_config=TrainingConfig.from_yaml_file(CONFIG_FILE_PATH),
)

if is_falsified:
    print("Verification problem is falsified.")
    sys.exit(0)

unstable_masks = solver.sequential.unstable_masks

compare_against_gurobi(
    new_L_list=[torch.from_numpy(x) for x in new_L_list],
    new_U_list=[torch.from_numpy(x) for x in new_U_list],
    unstable_masks=unstable_masks,
    initial_L_list=solver_inputs.L_list,
    initial_U_list=solver_inputs.U_list,
    gurobi_results=gurobi_results,
    cutoff_threshold=1e-5,
)
