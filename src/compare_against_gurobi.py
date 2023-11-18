from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .inputs.save_file_types import GurobiResults


def compare_against_gurobi(
    new_L: List[Tensor],
    new_U: List[Tensor],
    unstable_masks: List[Tensor],
    initial_L: List[Tensor],
    initial_U: List[Tensor],
    gurobi_results: GurobiResults,
    cutoff_threshold: Optional[float] = None,
) -> None:
    # Ensure all tensors are on same device.
    device = torch.device("cpu")
    new_L = [L_i.to(device) for L_i in new_L]
    new_U = [U_i.to(device) for U_i in new_U]
    unstable_masks = [mask.to(device) for mask in unstable_masks]
    initial_L = [L_i.to(device) for L_i in initial_L]
    initial_U = [U_i.to(device) for U_i in initial_U]
    gurobi_results["L_unstable_only"] = [
        L_i.to(device) for L_i in gurobi_results["L_unstable_only"]
    ]
    gurobi_results["U_unstable_only"] = [
        U_i.to(device) for U_i in gurobi_results["U_unstable_only"]
    ]

    # Only consider input + unstable intermediates neurons.
    masks = unstable_masks[1:-1]
    unstable_L = [initial_L[0]] + [L_i[mask] for (L_i, mask) in zip(initial_L[1:-1], masks)]
    unstable_U = [initial_U[0]] + [U_i[mask] for (U_i, mask) in zip(initial_U[1:-1], masks)]
    unstable_new_L = [new_L[0]] + [L_i[mask] for (L_i, mask) in zip(new_L[1:-1], masks)]
    unstable_new_U = [new_U[0]] + [U_i[mask] for (U_i, mask) in zip(new_U[1:-1], masks)]
    gurobi_L = gurobi_results["L_unstable_only"][:-1]
    gurobi_U = gurobi_results["U_unstable_only"][:-1]

    list_len: int = len(unstable_new_L)

    # Assert that all bounds lists are of same length/shape.
    assert (
        len(unstable_L)
        == len(unstable_U)
        == len(unstable_new_L)
        == len(unstable_new_U)
        == len(gurobi_L)
        == len(gurobi_U)
    )
    for i in range(list_len):
        assert (
            unstable_L[i].shape
            == unstable_U[i].shape
            == unstable_new_L[i].shape
            == unstable_new_U[i].shape
            == gurobi_L[i].shape
            == gurobi_U[i].shape
        )

    diff_L: List[Tensor] = [gurobi_L[i] - unstable_L[i] for i in range(list_len)]
    diff_U: List[Tensor] = [unstable_U[i] - gurobi_U[i] for i in range(list_len)]
    diff_new_L: List[Tensor] = [gurobi_L[i] - unstable_new_L[i] for i in range(list_len)]
    diff_new_U: List[Tensor] = [unstable_new_U[i] - gurobi_U[i] for i in range(list_len)]

    if cutoff_threshold:
        non_zero_L_mask: List[Tensor] = [(x > cutoff_threshold) for x in diff_L]
        non_zero_U_mask: List[Tensor] = [(x > cutoff_threshold) for x in diff_U]

        diff_L = [diff_L[i][non_zero_L_mask[i]] for i in range(list_len)]
        diff_U = [diff_U[i][non_zero_U_mask[i]] for i in range(list_len)]
        diff_new_L = [diff_new_L[i][non_zero_L_mask[i]] for i in range(list_len)]
        diff_new_U = [diff_new_U[i][non_zero_U_mask[i]] for i in range(list_len)]

    plot_box_and_whiskers(
        [diff_L, diff_U, diff_new_L, diff_new_U],
        ["initial L", "initial U", "new L", "new U"],
        title="Difference between computed bounds vs Gurobi's"
        + f"\n(excluding neurons whr initial-vs-Gurobi diff values <= {cutoff_threshold})",
        xlabel="Differences",
        ylabel="Bounds",
    )


def plot_box_and_whiskers(
    values: List[List[Tensor]],
    labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    concat_values: List[np.ndarray] = [torch.cat(x).numpy() for x in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(concat_values, vert=False, labels=labels)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()
