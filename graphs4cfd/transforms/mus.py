import torch
from torch_geometric.nn import voxel_grid
from torch_scatter import scatter_mean
from typing import Tuple, List

from ..graph import Graph


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