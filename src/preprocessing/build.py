from typing import Dict, Iterable, List, cast

import networkx as nx
from torch import fx, nn

from ..modules.solver_layers.add import Add_SL
from ..modules.solver_layers.input_layer import Input_SL
from ..modules.solver_layers.l1 import L1_SL
from ..modules.solver_layers.misc import Misc_SL
from ..modules.solver_layers.output_layer import Output_SL
from ..modules.solver_layers.relu import ReLU_SL
from ..preprocessing.graph_converters import (
    networkx_to_pytorch_graph,
    pytorch_graph_to_networkx,
)
from ..preprocessing.named_solver_inputs import NamedSolverInputs
from . import preprocessing_utils
from .solver_inputs import SolverInputs
from .transpose import transpose_layer


def map_modules(
    modules: Dict[str, nn.Module],
    named_solver_inputs: NamedSolverInputs,
) -> Dict[str, nn.Module]:
    num_batches: int = named_solver_inputs.C_dict["input_layer"].size(0)
    mapped_modules: Dict[str, nn.Module] = {
        "input_layer": Input_SL(
            L=named_solver_inputs.L_dict["input_layer"],
            U=named_solver_inputs.U_dict["input_layer"],
            C=named_solver_inputs.C_dict["input_layer"],
        ),
        "output_layer": Output_SL(
            num_batches=num_batches,
            H=named_solver_inputs.H,
            d=named_solver_inputs.d,
        ),
    }
    for name, module in modules.items():
        if isinstance(module, nn.BatchNorm2d):
            continue

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            input_shape = named_solver_inputs.input_shapes[name]
            output_shape = named_solver_inputs.output_shapes[name]
            transposed_layer, bias_module = transpose_layer(module, input_shape, output_shape)
            mapped_modules[name] = L1_SL(transposed_layer, bias_module)
            continue

        if isinstance(module, nn.ReLU):
            mapped_modules[name] = ReLU_SL(
                L=named_solver_inputs.L_dict[name],
                U=named_solver_inputs.U_dict[name],
                C=named_solver_inputs.C_dict[name],
                P=named_solver_inputs.P_dict[name],
                P_hat=named_solver_inputs.P_hat_dict[name],
                p=named_solver_inputs.p_dict[name],
            )
            continue

        if preprocessing_utils.is_add_layer(module):
            mapped_modules[name] = Add_SL()
            continue

        input_shape = named_solver_inputs.input_shapes[name]
        output_shape = named_solver_inputs.output_shapes[name]
        transposed_layer, _ = transpose_layer(module, input_shape, output_shape)
        mapped_modules[name] = Misc_SL(transposed_layer)

    return mapped_modules


def build_solver_graph_module(inputs: SolverInputs) -> fx.GraphModule:
    # fmt: off
    placeholder_nodes: List[fx.Node] = [x for x in inputs.model.graph.nodes if x.op == "placeholder"]
    assert len(placeholder_nodes) == 1, f"Expected model to have exactly 1 input, but got {len(placeholder_nodes)}."
    assert placeholder_nodes[0].target == 'input_1', f'Expected input node to be named "input_1", but got "{placeholder_nodes[0].target}"'
    # fmt: on

    model = inputs.model
    preprocessing_utils.freeze_model(model)
    DG = pytorch_graph_to_networkx(model.graph)
    DG = cast(nx.DiGraph, DG.reverse())
    DG = cast(
        nx.DiGraph,
        nx.relabel_nodes(DG, {"input_1": "input_layer", "output": "output_layer"}),
    )
    DG.add_node("output")
    DG.add_edge("input_layer", "output")

    # Remove `BatchNorm2d` layers.
    batch_norm_names = (name for name, module in model.named_children() if isinstance(module, nn.BatchNorm2d))  # fmt: skip
    for name in batch_norm_names:
        DG.add_edge(*DG.predecessors(name), *DG.successors(name))
        DG.remove_node(name)

    fx_graph = networkx_to_pytorch_graph(DG, has_input=False)

    def exclude_one_child_node(child_nodes: List[fx.Node]) -> List[fx.Node]:
        for node in child_nodes:
            if len(node.all_input_nodes) > 1:
                child_nodes.remove(node)
                return child_nodes
        return child_nodes[1:]

    add_names = (name for name, module in model.named_children() if preprocessing_utils.is_add_layer(module))  # fmt: skip
    for name in add_names:
        children_names = [child_name for _, child_name in DG.out_edges(name)]
        children_nodes = [x for x in cast(Iterable[fx.Node], fx_graph.nodes) if x.target in children_names]  # fmt: skip
        for child_node in exclude_one_child_node(children_nodes):
            child_node.kwargs = {"set_zero_accum": True}

    output_fx_node: fx.Node = next(x for x in model.graph.nodes if x.op == "output")
    last_module_fx_node = output_fx_node.all_input_nodes[0]
    output_module_name = cast(str, last_module_fx_node.target)

    unstable_masks = [(L < 0) & (U > 0) for L, U in zip(inputs.L_list, inputs.U_list)]

    # Initially set to solve for input layer.
    C_list, solve_coords = preprocessing_utils.get_C_for_layer(0, unstable_masks)

    submodules = {name: module for name, module in model.named_children()}
    named_solver_inputs = NamedSolverInputs(inputs, C_list)

    return fx.GraphModule(
        root=map_modules(submodules, named_solver_inputs),
        graph=fx_graph,
    )
