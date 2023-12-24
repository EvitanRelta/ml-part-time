from collections.abc import Iterator
from typing import Callable, Dict, Iterator, List, Tuple, TypeVar, Union

from torch import Tensor, fx, nn

from ..modules.solver_layers.input_layer import Input_SL
from ..modules.solver_layers.l1 import L1_SL
from ..modules.solver_layers.misc import Misc_SL
from ..modules.solver_layers.output_layer import Output_SL
from ..modules.solver_layers.relu import ReLU_SL
from ..modules.solver_layers.sum import Sum_SL
from ..preprocessing.graph_module_wrapper import GraphModuleWrapper, NodeWrapper
from ..preprocessing.named_solver_inputs import NamedSolverInputs
from . import preprocessing_utils
from .solver_inputs import SolverInputs
from .transpose import transpose_layer


def pick(tuple_index: int) -> Callable[[Tuple[Tensor, ...]], Tensor]:
    return lambda x: x[tuple_index]


def make_tuple(*args) -> Tuple[Tensor, ...]:
    return tuple(args)


def build_intermediate(
    node: NodeWrapper,
    prev_output: fx.Node,
    solver_modules: Dict[str, nn.Module],
    graph: fx.Graph,
    named_solver_inputs: NamedSolverInputs,
    out_args: List[fx.Node],
    relu_sum_nodes: Dict[str, fx.Node],
    relu_sum_nodes_args: Dict[str, List[fx.Node]],
    was_add_child: bool = False,
):
    if isinstance(node.module, (nn.Linear, nn.Conv2d)):
        transposed_layer, bias_module = transpose_layer(
            node.module,
            node.input_shape,
            node.output_shape,
        )
        l1_solver_layer = L1_SL(
            transposed_layer=transposed_layer,
            bias_module=bias_module,
        )

        prev_output = graph.call_module(node.name, (prev_output,))
        solver_modules[node.name] = l1_solver_layer

    elif isinstance(node.module, nn.ReLU):
        if len(node.children) > 1:
            if not node.name in relu_sum_nodes_args:
                relu_sum_nodes_args[node.name] = []

            sum_node_args = relu_sum_nodes_args[node.name]
            # If previous layer wasn't an Add layer, include in the Sum_SL layer's args.
            if not was_add_child:
                sum_node_args.append(prev_output)

            non_add_children = [
                x for x in node.children if not preprocessing_utils.is_add_layer(x.module)
            ]
            if len(non_add_children) != len(sum_node_args) or (node.name in relu_sum_nodes):
                return

            sum_node_name = node.name + "_sum"
            sum_solver_layer = Sum_SL()
            sum_node = graph.call_module(sum_node_name, tuple(sum_node_args))
            solver_modules[sum_node_name] = sum_solver_layer
            relu_sum_nodes[node.name] = sum_node
            prev_output = sum_node

        relu_solver_layer = ReLU_SL(
            L=named_solver_inputs.L_dict[node.name],
            U=named_solver_inputs.U_dict[node.name],
            C=named_solver_inputs.C_dict[node.name],
            P=named_solver_inputs.P_dict[node.name],
            P_hat=named_solver_inputs.P_hat_dict[node.name],
            p=named_solver_inputs.p_dict[node.name],
        )

        prev_output = graph.call_module(node.name, (prev_output,))
        solver_modules[node.name] = relu_solver_layer

    elif isinstance(node.module, nn.BatchNorm2d):
        pass
    elif preprocessing_utils.is_add_layer(node.module):
        for parent in node.parents:
            build_intermediate(
                parent,
                prev_output,
                solver_modules,
                graph,
                named_solver_inputs,
                out_args,
                relu_sum_nodes,
                relu_sum_nodes_args,
                was_add_child=True,
            )
        return
    else:
        transposed_layer, _ = transpose_layer(
            node.module,
            node.input_shape,
            node.output_shape,
        )
        others_solver_layer = Misc_SL(transposed_layer)
        prev_output = graph.call_module(node.name, (prev_output,))
        solver_modules[node.name] = others_solver_layer

    if len(node.parents) == 0:
        out_args.append(prev_output)
        return

    for parent in node.parents:
        build_intermediate(
            parent,
            prev_output,
            solver_modules,
            graph,
            named_solver_inputs,
            out_args,
            relu_sum_nodes,
            relu_sum_nodes_args,
        )


def build_solver_graph_module(inputs: SolverInputs) -> fx.GraphModule:
    preprocessing_utils.freeze_model(inputs.model)

    graph_wrapper = GraphModuleWrapper(inputs.model, inputs.input_shape)
    unstable_masks = [(L < 0) & (U > 0) for L, U in zip(inputs.L_list, inputs.U_list)]

    # Initially set to solve for input layer.
    C_list, solve_coords = preprocessing_utils.get_C_for_layer(0, unstable_masks)

    named_solver_inputs = NamedSolverInputs(inputs, C_list)

    last_node = graph_wrapper.last_child
    solver_modules: Dict[str, nn.Module] = {}
    graph = fx.Graph()

    transposed_layer, bias_module = transpose_layer(
        last_node.module,
        last_node.input_shape,
        last_node.output_shape,
    )
    output_layer = Output_SL(
        transposed_layer=transposed_layer,
        bias_module=bias_module,
        L=named_solver_inputs.L_dict["output_layer"],
        U=named_solver_inputs.U_dict["output_layer"],
        C=named_solver_inputs.C_dict["output_layer"],
        H=inputs.H,
        d=inputs.d,
    )
    solver_modules["output_layer"] = output_layer
    prev_output = graph.call_module("output_layer")

    out_args: List[fx.Node] = []
    relu_sum_nodes: Dict[str, fx.Node] = {}
    relu_sum_nodes_args: Dict[str, List[fx.Node]] = {}
    for parent in last_node.parents:
        build_intermediate(
            parent,
            prev_output,
            solver_modules,
            graph,
            named_solver_inputs,
            out_args,
            relu_sum_nodes,
            relu_sum_nodes_args,
        )

    assert len(out_args) == 1
    prev_output = out_args[0]

    solver_modules["input_layer"] = Input_SL(
        L=named_solver_inputs.L_dict["input_layer"],
        U=named_solver_inputs.U_dict["input_layer"],
        C=named_solver_inputs.C_dict["input_layer"],
    )
    prev_output = graph.call_module("input_layer", (prev_output,))
    graph.output(prev_output)

    return fx.GraphModule(solver_modules, graph)


T = TypeVar("T")


def get_reversed_iterator(list_or_iterator: Union[List[T], Iterator[T]]) -> Iterator[T]:
    items = list(list_or_iterator)
    items.reverse()
    return iter(items)
