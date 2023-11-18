# Setup

- Install Python >=3.8. \
  If you're using Anaconda, create a new environment via:

  ```bash
  conda create -n your_env python=3.8
  conda activate your_env
  ```

- Install Pytorch >=2.0.1 with CUDA from: \
  https://pytorch.org/get-started

  or if your CUDA version isn't listed, try from here: \
  https://pytorch.org/get-started/previous-versions

- Install the other dependencies in `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

<br>

# Running

## Solving toy example

Run the `train_toy.py` script:

```bash
# In the repo's root:
python train_toy.py
```

<br>

## Solving MNIST 256x6 & comparing against Gurobi's

Run the script at `train_mnist_256x6.py` via:

```bash
# In the repo's root:
python train_mnist_256x6.py
```

<br>

## Solving ConvMed
NOT IMPLEMENTED YET

<br>

## General solving + visualising code

```py
#====================================================================
#                Preparing all the inputs for solving.
#====================================================================

import os
from torch import nn
from src.inputs.utils import set_abs_path_to
from src.utils import load_onnx_model

# Loading ONNX model.
try:
    CURRENT_DIR = os.path.dirname(__file__)  # For `.py` files.
except:
    CURRENT_DIR = os.getcwd()  # For `.ipynb` files.
get_abs_path = set_abs_path_to(CURRENT_DIR)

absolute_model_path = get_abs_path("./path/to/model.onnx")
model: nn.Module = load_onnx_model(absolute_model_path)


from typing import List
from torch import Tensor
from src.preprocessing.solver_inputs import SolverInputs

# Other solver inputs.
ground_truth_neuron_index: int
L: List[Tensor]
U: List[Tensor]
H: Tensor
d: Tensor
P: List[Tensor]
P_hat: List[Tensor]
p: List[Tensor]

solver_inputs = SolverInputs(
    model=model,
    ground_truth_neuron_index=ground_truth_neuron_index,
    L=L,
    U=U,
    H=H,
    d=d,
    P=P,
    P_hat=P_hat,
    p=p,
)


#====================================================================
#                              Solving
#====================================================================
import torch
from src.utils import seed_everything
from src.solve import solve

# For deterministic results.
seed_everything(0)

# device = torch.device("cpu")  # solve on CPU
device = torch.device("cuda")  # solve on GPU

is_falsified, new_lower_bounds, new_upper_bounds = \
    solve(solver_inputs, device=device)


#====================================================================
#                    Comparing results to Gurobi's
#====================================================================
from typing import List
from torch import Tensor
from src.compare_against_gurobi import compare_against_gurobi
from src.inputs.save_file_types import GurobiResults

initial_L: List[Tensor] = L
initial_U: List[Tensor] = U
gurobi_lower_bounds_unstable_only: List[Tensor]
gurobi_upper_bounds_unstable_only: List[Tensor]
gurobi_compute_time: float

gurobi_results: GurobiResults = {
    "L_unstable_only": gurobi_lower_bounds_unstable_only,
    "U_unstable_only": gurobi_upper_bounds_unstable_only,
    "compute_time": gurobi_compute_time,
}

unstable_masks = [
    (initial_L[i] < 0) & (initial_U[i] > 0) \
        for i in range(len(initial_L))
]

compare_against_gurobi(
    new_L=new_lower_bounds,
    new_U=new_upper_bounds,
    initial_L=initial_L,
    initial_U=initial_U,
    unstable_masks=unstable_masks,
    gurobi_results=gurobi_results,
)
```
