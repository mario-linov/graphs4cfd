import random
from typing import Iterable, Union

from ..graph import Graph


class NodeSubset():
    """Transformation class for selecting a subset of nodes from a graph.

    Args:
        idx (Iterable[int]): Indices of nodes to keep.

    Methods:
        __call__(graph: Graph) -> Graph: Selects a subset of nodes from a graph.
    """

    def __init__(self, idx: Iterable[int]):
        self.idx = idx

    def __call__(self, graph: Graph) -> Graph:
        idx = self.idx
        graph.pos    = graph.pos  [idx]
        graph.field  = graph.field[idx]
        if hasattr(graph, 'omega' ): graph.omega  = graph.omega [idx]
        if hasattr(graph, 'target'): graph.target = graph.target[idx]
        if hasattr(graph, 'bound' ): graph.bound  = graph.bound [idx]
        if hasattr(graph, 'loc'   ): graph.loc    = graph.loc   [idx]
        if hasattr(graph, 'glob'  ): graph.glob   = graph.glob  [idx]
        return graph


class RandomNodeSubset():
    """Transformation class for randomly selecting a subset of nodes from a graph.
    
    Args:
        num_nodes (Union[float, int]): Number of nodes to keep. If float, interpret as percentage.
            If int, interpret as number of nodes.

    
    Methods:
        __call__(graph: Graph) -> Graph: Randomly selects a subset of nodes from a graph.
    """

    def __init__(self, num_nodes: Union[float, int]):
        self.num_nodes = num_nodes

    def __call__(self, graph: Graph) -> Graph:
        # Determine indices of nodes to keep
        if isinstance(self.num_nodes, float): # If float, interpret as percentage
            idx = random.sample(range(graph.num_nodes), k=int(self.num_nodes*graph.num_nodes))
        else: # If int, interpret as number of nodes
            idx = random.sample(range(graph.num_nodes), k=self.num_nodes)
        # Subset graph
        graph.pos    = graph.pos  [idx]
        graph.field  = graph.field[idx]
        if hasattr(graph, 'omega' ): graph.omega  = graph.omega [idx]
        if hasattr(graph, 'target'): graph.target = graph.target[idx]
        if hasattr(graph, 'bound' ): graph.bound  = graph.bound [idx]
        if hasattr(graph, 'loc'   ): graph.loc    = graph.loc   [idx]
        if hasattr(graph, 'glob'  ): graph.glob   = graph.glob  [idx]
        return graph