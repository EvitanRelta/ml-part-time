from collections import OrderedDict
from typing import Iterable, Iterator, cast

import networkx as nx
from torch import fx


def pytorch_graph_to_networkx(graph: fx.Graph) -> nx.DiGraph:
    """Converts a `torch.fx.Graph` to a `networkx` directed-graph."""
    DG = nx.DiGraph()
    nodes = cast(Iterable[fx.Node], graph.nodes)
    for node in nodes:
        DG.add_node(node.target)
        for child in node.users:
            DG.add_edge(node.target, child.target)
    return DG


def networkx_to_pytorch_graph(DG: nx.DiGraph, has_input: bool = True) -> fx.Graph:
    """Converts a `networkx` directed-graph to a `torch.fx.Graph`.

    Assumes:
    - model has either EXACTLY 1 input (requiring `has_input=True`)
    or no inputs (requiring `has_input=False`). Does not support >1 input.
    - graph only contains
    - each node in `DG` is either an input, output, or module node. No function
      nodes.
    - nodes in `DG` that has >1 in-edge represents modules that takes in >1
      positional args (eg. `(arg1, arg2)` for a node with 2 in-edges).

    Args:
        DG (networkx.DiGraph): The directed-graph to convert from.
        has_input (bool, optional): Whether the model has an input. Defaults to True.

    Returns:
        torch.fx.Graph: The converted PyTorch graph.
    """
    created_fx_nodes: OrderedDict[str, fx.Node] = OrderedDict()
    fx_graph = fx.Graph()
    nodes_names = cast(Iterator[str], nx.topological_sort(DG))

    # Has exactly 1 input.
    if has_input:
        # First node is the input.
        input_node_name = next(nodes_names)
        fx_node = fx_graph.placeholder(input_node_name)
        created_fx_nodes[input_node_name] = fx_node

    nodes_names = list(nodes_names)

    # Iterate over all nodes except the last, which is the output node.
    for node_name in nodes_names[:-1]:
        parent_names = (parent_name for parent_name, _ in DG.in_edges(node_name))
        parent_fx_nodes = (created_fx_nodes[name] for name in parent_names)
        fx_node = fx_graph.call_module(node_name, tuple(parent_fx_nodes))
        created_fx_nodes[node_name] = fx_node

    last_module_fx_node = next(reversed(created_fx_nodes.values()))
    fx_graph.output(last_module_fx_node)
    return fx_graph
