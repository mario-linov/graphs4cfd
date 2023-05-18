import torch
from typing import Tuple, List, Union, Optional

from ..graph import Graph
from .connect import connect_knn

    
def guillard_coarsening(edge_index: torch.Tensor,
                        num_nodes: int) -> torch.Tensor:
        """Modified version of the node-nested coarsening by Guillard (https://hal.science/inria-00074773/).
        It is used to create the low-resolution graphs of gMuS-GNNs and REMuS-GNNs.
        It assumes that the indegree of every node is the same.
        
        Args:
            edge_index (torch.Tensor): Edge index of the graph.
            num_nodes (int): Number of nodes of the graph.
        
        Returns:
            torch.Tensor: Coarse mask. It is a boolean tensor of size num_nodes.        
        """
        # Determine the indegree
        k = (edge_index[1]==0).sum().item()
        # Find senders
        senders = edge_index[0].view(-1,k)
        # Node-nested coarsening by Guillard
        coarse_mask = torch.ones(num_nodes, dtype=torch.bool)
        for coarse_node, s in zip(coarse_mask, senders):
            if coarse_node: coarse_mask[s] = False
        return coarse_mask
    

class GuillardCoarseningAndConnectKNN():
    r""" Transformation class that coarsens a graph using the node-nested coarsening by Guillard (https://hal.science/inria-00074773/).
    It is used to create the low-resolution graphs of gMuS-GNNs.
    It assumes that the indegree of every node is the same.
    It also connects the nodes of the graph using the k-nearest neighbours algorithm.

    Args:
        k (List[int]): Number of neighbours to connect at each level. The number of elements in the list determines the number of low-resolution graphs in the gMuS-GNN.
            At the i-th level, the number of neighbours is k[i-1].
        period (Optional[Union[None,Tuple]]): Period of the grid. If None, the grid is not periodic. If a tuple, it is the period of the grid.
        scale_edge_attr (Optional[Union[None, Tuple]]): Scale of the edge attributes. If None, the edge attributes are not scaled. If a tuple, it is the scale of the edge attributes.

    Methods:
        __call__(graph: Graph) -> Graph: Coarsens a graph using the node-nested coarsening by Guillard (https://hal.science/inria-00074773/).
    """
    
    def __init__(self, 
                 k: List[int],
                 period: Optional[Union[None,Tuple]] = None,
                 scale_edge_attr: Optional[Union[None, Tuple]] = None):
        assert len(k) > 1 and len(k) < 5, "The number of levels in gMuS-GNN must be between 2 and 4."
        self.k = k
        self.period = period
        self.scale_edge_attr = scale_edge_attr

    def __call__(self, graph: Graph) -> Graph:
        num_levels = len(self.k) # Number of levels in gMuS-GNN
        # Connect level 1
        graph.edge_index, graph.edge_attr = connect_knn(graph.pos, self.k[0], period=self.period)
        # Coarsen level 1
        graph.coarse_mask2 = guillard_coarsening(graph.edge_index, graph.num_nodes) # Mask applied to V^1 to obtain V^2
        coarse_index2 = graph.coarse_mask2.nonzero().squeeze() # V^1-index of the nodes in V^2
        # Connect level 2
        graph.edge_index2, graph.edge_attr2 = connect_knn(graph.pos[coarse_index2], self.k[1], period=self.period)
        if num_levels > 2:
            # Coarsen level 2
            graph.coarse_mask3 = torch.zeros_like(graph.coarse_mask2, dtype=torch.bool)
            graph.coarse_mask3[graph.coarse_mask2] = self.guillard_coarsening(graph.edge_index2, graph.coarse_mask2.sum()) # Mask applied to V^1 to obtain V^3
            coarse_index3 = graph.coarse_mask3.nonzero().squeeze() # V^1-index of the nodes in V^3
            # Connect level 3
            graph.edge_index3, graph.edge_attr3 = connect_knn(graph.pos[coarse_index3], self.k[2], period=self.period)
            if num_levels > 3:
                # Coarsen level 3
                graph.coarse_mask4 = torch.zeros_like(graph.coarse_mask3, dtype=torch.bool)
                graph.coarse_mask4[graph.coarse_mask3] = self.guillard_coarsening(graph.edge_index3, graph.coarse_mask3.sum()) # Mask applied to V^1 to obtain V^4
                coarse_index4 = graph.coarse_mask4.nonzero().squeeze() # V^1-index of the nodes in V^4
                # Connect level 4
                graph.edge_index4, graph.edge_attr4 = connect_knn(graph.pos[coarse_index4], self.k[3], period=self.period)
        # Renumber edge index to the original indices
        graph.edge_index2 = coarse_index2[graph.edge_index2]
        if num_levels > 2: graph.edge_index3 = coarse_index3[graph.edge_index3]
        if num_levels > 3: graph.edge_index4 = coarse_index4[graph.edge_index4]
        # Scale both edge_attr
        if self.scale_edge_attr[0] is not None: graph.edge_attr  /= (2*self.scale_edge_attr[0])
        if self.scale_edge_attr[1] is not None: graph.edge_attr2 /= (2*self.scale_edge_attr[1])
        if num_levels > 2 and self.scale_edge_attr[2] is not None: graph.edge_attr3 /= (2*self.scale_edge_attr[2])
        if num_levels > 3 and self.scale_edge_attr[3] is not None: graph.edge_attr4 /= (2*self.scale_edge_attr[3])
        return graph