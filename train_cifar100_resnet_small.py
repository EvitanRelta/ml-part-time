import os
import sys

import torch

from src.inputs.cifar100_resnet_small import solver_inputs
from src.solve import solve
from src.training.TrainingConfig import TrainingConfig
from src.utils import seed_everything, set_abs_path_to

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
CONFIG_FILE_PATH = get_abs_path("default_training_config.yaml")

seed_everything(0)

# Load training config from YAML file.
training_config = TrainingConfig.from_yaml_file(CONFIG_FILE_PATH)
training_config.disable_adv_check = True

is_falsified, new_L_list, new_U_list, solver = solve(
    solver_inputs,
    device=torch.device("cpu"),
    return_solver=True,
    training_config=training_config,
)

if is_falsified:
    print("Verification problem is falsified.")
    sys.exit(0)
