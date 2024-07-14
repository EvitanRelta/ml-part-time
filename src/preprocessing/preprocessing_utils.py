import itertools
from typing import Iterable, Iterator, List, Tuple, cast

import torch
from onnx2torch.node_converters import (
    OnnxBinaryMathOperation,
    OnnxConstant,
    OnnxReshape,
)
from torch import Tensor, fx, nn
from typing_extensions import TypeAlias


def is_add_layer(module: nn.Module) -> bool:
    return isinstance(module, OnnxBinaryMathOperation) and module.math_op_function is torch.add


def freeze_model(model: nn.Module) -> None:
    """Freezes the model's learnable parameters."""
    for param in model.parameters():
        param.requires_grad = False


def remove_first_n_modules(graph_module: fx.GraphModule, n: int) -> None:
    """Mutably remove the first `n` number of modules from a
    `torch.fx.GraphModule`.

    Args:
        graph_module (fx.GraphModule): `GraphModule` to remove the layers from.
        n (int): Number of modules to remove.
    """
    nodes = cast(Iterator[fx.Node], iter(graph_module.graph.nodes))
    input_node = next(nodes)  # Pop the input node, as we don't include that in the removal
    nodes_to_remove = list(itertools.islice(nodes, n))
    assert all(
        len(node.users) == 1 for node in nodes_to_remove
    ), "Failed assumption that all nodes to remove only has 1 user."

    # Find the node that will be the new first node after the removal.
    new_first_node = next(iter(nodes_to_remove[-1].users))

    # Replace the argument of the first node after removal with the input node.
    new_first_node.args = (input_node,)

    # Remove the nodes, starting from the back.
    for node in reversed(nodes_to_remove):
        graph_module.graph.erase_node(node)

    # Recompile the graph
    graph_module.recompile()
    graph_module.delete_all_unused_submodules()


NeuronCoords: TypeAlias = Tuple[int, Tuple[int, ...]]
"""Coordinates for a neuron in the model, in the form `(layer_index, neuron_index_tuple)`."""


def get_C_for_layer(
    layer_index: int,
    unstable_masks: List[Tensor],
) -> Tuple[List[Tensor], List[NeuronCoords]]:
    """Get the `C_list` to solve for the unstable neurons in layer `layer_index`,
    where `layer_index` can be any layer except the last (as we don't solve for
    output layer).

    If `layer_index == 0`, `C_list` will solve all inputs neurons (irregardless of
    whether they're unstable).
    """
    device = unstable_masks[0].device
    num_layers = len(unstable_masks)
    assert layer_index < num_layers

    C_list: List[Tensor] = []
    coords: List[NeuronCoords] = []

    # For input layer, solve for all input neurons.
    if layer_index == 0:
        mask = unstable_masks[0]
        num_input_neurons = mask.numel()
        C_0 = torch.zeros((num_input_neurons * 2, *mask.shape)).to(device)
        batch_index: int = 0

        # Generating indices for each element.
        element_indices = itertools.product(*[range(size) for size in mask.shape])
        for index in element_indices:
            C_0[batch_index][index] = 1  # Minimising
            C_0[batch_index + 1][index] = -1  # Maximising
            batch_index += 2
            coords.append((0, index))

        C_list.append(C_0)
        for i in range(1, num_layers):
            mask: Tensor = unstable_masks[i]
            num_neurons: int = len(mask)
            C_list.append(torch.zeros((num_input_neurons * 2, *mask.shape)).to(device))
        return C_list, coords

    # Else, solve for only unstable neurons in the specified layer.
    num_unstable_in_target_layer = int(unstable_masks[layer_index].sum().item())
    for i in range(num_layers):
        mask: Tensor = unstable_masks[i]
        if i != layer_index:
            C_list.append(torch.zeros((num_unstable_in_target_layer * 2, *mask.shape)).to(device))
            continue

        unstable_indices = torch.nonzero(mask)
        C = torch.zeros((num_unstable_in_target_layer * 2, *mask.shape)).to(device)
        batch_index: int = 0
        for index in unstable_indices:
            index = tuple(index.tolist())
            C[batch_index][index] = 1  # Minimising
            C[batch_index + 1][index] = -1  # Maximising
            batch_index += 2
            coords.append((i, index))
        C_list.append(C)
    return C_list, coords


def replace_reshape_with_flatten(model: fx.GraphModule) -> fx.GraphModule:
    """Destructively replace all `Reshape` layers in `model` with `torch.nn.Flatten`.

    This is for ONNX models that has `Reshape` layers inplace of `Flatten` which
    causes problems for our solver, and thus needs to be replaced.
    """
    modules = {name: module for name, module in model.named_children()}
    graph = model.graph

    # The reshape layers will have an module named `initializers` which needs to
    # be removed.

    # Remove the reshapes' references to the `initializers` module, and replace
    # each reshape layer with `torch.nn.Flatten` layer.
    for node in cast(Iterable[fx.Node], graph.nodes):
        if (
            node.op == "call_module"
            and isinstance(node.target, str)
            and isinstance(modules[node.target], OnnxReshape)
        ):
            node.args = (node.args[0],)
            modules[node.target] = nn.Flatten()

    # Finally remove the artifact `initializers` module.
    for node in cast(Iterable[fx.Node], graph.nodes):
        if (
            node.op == "get_attr"
            and isinstance(node.target, str)
            and node.target.startswith("initializers")
        ):
            graph.erase_node(node)

    return fx.GraphModule(modules, graph)


def remove_onnx_norm_layers(model: fx.GraphModule) -> None:
    """Mutably remove the ONNX normalization layers (if any) in `model`."""
    modules = {name: module for name, module in model.named_children()}
    graph = model.graph

    nodes = cast(Iterator[fx.Node], iter(graph.nodes))
    input_node = next(nodes)
    nodes_to_remove: list[fx.Node] = []
    for node in nodes:
        if node.op != "call_module" or not isinstance(node.target, str):
            continue

        module = modules[node.target]
        if isinstance(module, (OnnxConstant, OnnxBinaryMathOperation)):
            nodes_to_remove.append(node)
            continue

        # Stop at first node that's not `OnnxConstant` or `OnnxBinaryMathOperation`.
        if len(nodes_to_remove) == 0:
            # If no nodes to remove, that means there's no norm layers, and thus
            # must take in input node.
            assert node.args[0] == input_node
            return

        assert len(node.args) == 1, "I'm assuming there's only 1 arg."
        node.args = (input_node,)
        break

    # Remove in reversed order to avoid removing dependencies b4 dependent nodes.
    for node in reversed(nodes_to_remove):
        graph.erase_node(node)

    # Recompile the graph
    model.recompile()
    model.delete_all_unused_submodules()
