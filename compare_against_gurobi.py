import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from inputs.save_file_types import GurobiResults


def compare_against_gurobi(
    new_L: list[Tensor],
    new_U: list[Tensor],
    unstable_masks: list[Tensor],
    initial_L: list[Tensor],
    initial_U: list[Tensor],
    gurobi_results: GurobiResults,
    cutoff_threshold: float | None = None,
) -> None:
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

    diff_L: list[Tensor] = [gurobi_L[i] - unstable_L[i] for i in range(list_len)]
    diff_U: list[Tensor] = [unstable_U[i] - gurobi_U[i] for i in range(list_len)]
    diff_new_L: list[Tensor] = [gurobi_L[i] - unstable_new_L[i] for i in range(list_len)]
    diff_new_U: list[Tensor] = [unstable_new_U[i] - gurobi_U[i] for i in range(list_len)]

    if cutoff_threshold:
        non_zero_L_mask: list[Tensor] = [(x > cutoff_threshold) for x in diff_L]
        non_zero_U_mask: list[Tensor] = [(x > cutoff_threshold) for x in diff_U]

        diff_L = [diff_L[i][non_zero_L_mask[i]] for i in range(list_len)]
        diff_U = [diff_U[i][non_zero_U_mask[i]] for i in range(list_len)]
        diff_new_L = [diff_new_L[i][non_zero_L_mask[i]] for i in range(list_len)]
        diff_new_U = [diff_new_U[i][non_zero_U_mask[i]] for i in range(list_len)]

    plot_box_and_whiskers(
        [diff_L, diff_U, diff_new_L, diff_new_U],
        ["initial L", "initial U", "new L", "new U"],
    )


def plot_box_and_whiskers(values: list[list[Tensor]], labels: list[str]) -> None:
    concat_values: list[np.ndarray] = [torch.cat(x).numpy() for x in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(concat_values, vert=False, labels=labels)

    ax.set_title("Horizontal Box Plots for diff between compute & gurobi bounds")
    ax.set_xlabel("Differences")
    ax.set_ylabel("Bounds")

    plt.tight_layout()
    plt.show()
