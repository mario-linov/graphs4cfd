import torch
from torch_geometric.nn import voxel_grid
from torch_scatter import scatter_mean
from typing import Tuple, List, Union, Optional

from ..graph import Graph
from .connect import connect_knn


def grid_clustering(pos_1: torch.Tensor, cell_size_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Clusters the nodes of a graph into cells of a grid. The nodes are assigned to the cell in which they are located.
        This is the algorithm for creating the low-resolution graphs of MuS-GNNs.
        
        Args:
            pos_1 (torch.Tensor): Node positions.
            cell_size_2 (torch.Tensor): Size of the cells of the grid.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                New node positions, cluster index of each node, cluster index of non-empty clusters,
                non-empyt-clusters index of each node, normalised relative position of each node with respect to the cell in which it is located.        
        """
        device = pos_1.device
        num_nodes = pos_1.size(0)
        # Create cells
        cluster_2 = voxel_grid(pos=pos_1, size=cell_size_2, batch=torch.zeros(num_nodes))
        # Find non-empty clusters
        mask_2, idx = cluster_2.unique().sort() # Map each cell "mask" to their new index "idx"
        num_clusters = mask_2.max().item()+1 # Number of empty and non-empty clusters/cells
        mask2idx = -torch.ones(num_clusters, device=device, dtype=torch.long) # Allows to get new index from cluster index, -1 means that the given cluster is empty 
        mask2idx[mask_2] = idx
        idx1_to_idx2 = mask2idx[cluster_2] # Lookup table that maps the V^1 index of each node to the V^2 index of its parent node
        # Compute new pos with new cluster indexing
        pos_2 = scatter_mean(pos_1, cluster_2, dim=0)[mask_2]
        # Realtive position 
        e_12  = pos_2[idx1_to_idx2] - pos_1
        e_12 /= cell_size_2
        return pos_2, cluster_2, mask_2, idx1_to_idx2, e_12

    
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


class GridClustering():
    """Clusters the nodes of a graph into cells of a grid. The nodes are assigned to the cell in which they are located.
    This is the algorithm for creating the low-resolution graphs of MuS-GNNs.

    Args:
        cells_size (List[float]): Size of the cells of the grid. The number of elements in the list determines the number of low-resolution graphs in the MuS-GNN.
            At the i-th level, the size of the cells is cells_size[i-1] x cells_size[i-1].

    Methods:
        __call__(graph: Graph) -> Graph: Clusters the nodes of a graph into cells of a grid. The nodes are assigned to the cell in which they are located.    
    """

    def __init__(self, cells_size: List[float]):
        self.num_levels = len(cells_size)+1 # Number of levels in MuS-GNN
        self.cells_size = cells_size

    def __call__(self, graph: Graph) -> Graph:
        # Create V^2
        graph.pos_2, graph.cluster_2, graph.mask_2, graph.idx1_to_idx2, graph.e_12 = grid_clustering(graph.pos, self.cells_size[0])
        if self.num_levels > 2:
            # Create V^3
            graph.pos_3, graph.cluster_3, graph.mask_3, graph.idx2_to_idx3, graph.e_23 = grid_clustering(graph.pos_2, self.cells_size[1])
        if self.num_levels > 3:
            # Create V^3
            graph.pos_4, graph.cluster_4, graph.mask_4, graph.idx3_to_idx4, graph.e_34 = grid_clustering(graph.pos_3, self.cells_size[2])
        return graph
    

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