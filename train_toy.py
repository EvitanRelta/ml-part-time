import os

import torch

from src.inputs.toy_example import solver_inputs
from src.solve import solve
from src.training.TrainingConfig import TrainingConfig
from src.utils import seed_everything, set_abs_path_to

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
CONFIG_FILE_PATH = get_abs_path("default_training_config.yaml")

seed_everything(0)

# Loading training configs from YAML file.
training_config = TrainingConfig.from_yaml_file(CONFIG_FILE_PATH)
assert not isinstance(training_config, list)

solve(solver_inputs, device=torch.device("cpu"), training_config=training_config)

print("Training finished!")
