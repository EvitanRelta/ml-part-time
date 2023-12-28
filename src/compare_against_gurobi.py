from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .inputs.save_file_types import GurobiResults


def compare_against_gurobi(
    new_L_list: List[Tensor],
    new_U_list: List[Tensor],
    unstable_masks: List[Tensor],
    initial_L_list: List[Tensor],
    initial_U_list: List[Tensor],
    gurobi_results: GurobiResults,
    cutoff_threshold: Optional[float] = None,
) -> None:
    """Plots a box-and-whisker plot of the new bounds against Gurobi's results.

    You can use `cutoff_threshold` to exclude the neurons where the initial
    bounds were already very close to the Gurobi-computed bounds.

    Args:
        new_L_list (List[Tensor]): New lower bounds.
        new_U_list (List[Tensor]): New upper bounds.
        unstable_masks (List[Tensor]): Masks of all the unstable neurons.
        initial_L_list (List[Tensor]): Initial lower bounds.
        initial_U_list (List[Tensor]): Initial upper bounds.
        gurobi_results (GurobiResults): Gurobi's results.
        cutoff_threshold (Optional[float], optional): If specified, excludes \
            neurons whr initial-vs-Gurobi absolute-difference values are
            <= `cutoff_threshold`. Defaults to None.
    """
    # Ensure all tensors are on same device.
    device = torch.device("cpu")
    new_L_list = [L.to(device) for L in new_L_list]
    new_U_list = [U.to(device) for U in new_U_list]
    unstable_masks = [mask.to(device) for mask in unstable_masks]
    initial_L_list = [L.to(device) for L in initial_L_list]
    initial_U_list = [U.to(device) for U in initial_U_list]
    gurobi_L_list = [L.to(device) for L in gurobi_results["L_list_unstable_only"]]
    gurobi_U_list = [U.to(device) for U in gurobi_results["U_list_unstable_only"]]

    # Remove output bounds, as it doesn't change.
    gurobi_L_list = gurobi_L_list[:-1]
    gurobi_U_list = gurobi_U_list[:-1]

    # Ensure bounds are flattened.
    new_L_list = [L.flatten() for L in new_L_list]
    new_U_list = [U.flatten() for U in new_U_list]
    unstable_masks = [mask.flatten() for mask in unstable_masks]
    initial_L_list = [L.flatten() for L in initial_L_list]
    initial_U_list = [U.flatten() for U in initial_U_list]

    # Only consider input + unstable intermediates neurons.
    masks = unstable_masks[1:]
    unstable_L_list = [initial_L_list[0]] + [
        L[mask] for (L, mask) in zip(initial_L_list[1:], masks)
    ]
    unstable_U_list = [initial_U_list[0]] + [
        U[mask] for (U, mask) in zip(initial_U_list[1:], masks)
    ]
    unstable_new_L_list = [new_L_list[0]] + [L[mask] for (L, mask) in zip(new_L_list[1:], masks)]
    unstable_new_U_list = [new_U_list[0]] + [U[mask] for (U, mask) in zip(new_U_list[1:], masks)]

    list_len: int = len(unstable_new_L_list)

    # Assert that all bounds lists are of same length/shape.
    assert (
        len(unstable_L_list)
        == len(unstable_U_list)
        == len(unstable_new_L_list)
        == len(unstable_new_U_list)
        == len(gurobi_L_list)
        == len(gurobi_U_list)
    )
    for i in range(list_len):
        assert (
            unstable_L_list[i].shape
            == unstable_U_list[i].shape
            == unstable_new_L_list[i].shape
            == unstable_new_U_list[i].shape
            == gurobi_L_list[i].shape
            == gurobi_U_list[i].shape
        )

    diff_L_list: List[Tensor] = [gurobi_L_list[i] - unstable_L_list[i] for i in range(list_len)]
    diff_U_list: List[Tensor] = [unstable_U_list[i] - gurobi_U_list[i] for i in range(list_len)]
    diff_new_L_list: List[Tensor] = [
        gurobi_L_list[i] - unstable_new_L_list[i] for i in range(list_len)
    ]
    diff_new_U_list: List[Tensor] = [
        unstable_new_U_list[i] - gurobi_U_list[i] for i in range(list_len)
    ]

    if cutoff_threshold:
        non_zero_L_mask: List[Tensor] = [(x.abs() > cutoff_threshold) for x in diff_L_list]
        non_zero_U_mask: List[Tensor] = [(x.abs() > cutoff_threshold) for x in diff_U_list]

        diff_L_list = [diff_L_list[i][non_zero_L_mask[i]] for i in range(list_len)]
        diff_U_list = [diff_U_list[i][non_zero_U_mask[i]] for i in range(list_len)]
        diff_new_L_list = [diff_new_L_list[i][non_zero_L_mask[i]] for i in range(list_len)]
        diff_new_U_list = [diff_new_U_list[i][non_zero_U_mask[i]] for i in range(list_len)]

    plot_box_and_whiskers(
        [diff_L_list, diff_U_list, diff_new_L_list, diff_new_U_list],
        ["initial lower bounds", "initial upper bounds", "new lower bounds", "new upper bounds"],
        title="Difference between computed bounds vs Gurobi's"
        + (
            f"\n(excluding neurons whr initial-vs-Gurobi abs-diff values <= {cutoff_threshold})"
            if cutoff_threshold is not None
            else ""
        ),
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
    """Plots multiple box-and-whisker diagrams in a single matplotlib plot.

    Args:
        values (List[List[Tensor]]): Values for each box-and-whisker diagram.
        labels (List[str]): Labels for each box-and-whisker diagram.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
    """
    concat_values: List[np.ndarray] = [torch.cat(x).numpy() for x in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(concat_values, vert=False, labels=labels)  # type: ignore

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.show()
