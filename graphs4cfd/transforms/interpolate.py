import os
import torch
import random
import numpy as np
from scipy.interpolate import griddata
from xml.etree import ElementTree
from typing import Union, Optional, Tuple
from torch_geometric.nn import knn


from ..graph import Graph


def interpolate_nodes(graph: Graph,
                      pos: torch.Tensor,
                      method: Optional[Union[str, None]] = None) -> Graph:
    r"""Interpolates the fields of a graph to a new set of nodes.
    
    Args:
        graph (Graph): Graph to interpolate.
        pos (torch.Tensor): New node positions.
        method (Optional[Union[str, None]], optional): Interpolation method. If `None`, the method is set to
            `'cubic'` if the dimension of the problem is 2, and `'linear'` otherwise. Defaults to `None`.

    Returns:
        Graph: Interpolated graph.

    Raises:
        ValueError: If the graph has edges.
    """
    # Check if graph is a set of nodes
    if graph.edge_index is not None:
        raise ValueError("Graphs cannot be interpolated, only sets of nodes.")
    # Dimension of the problem
    dim = pos.size(1)
    # Select interpolation method
    if method is None:
        method = 'cubic' if dim == 2 else 'linear'
    # Interpolate fields
    if hasattr(graph, 'loc' ): graph.loc  = torch.tensor( griddata(graph.pos, graph.loc,  pos, method=method).astype(np.float32) )
    if hasattr(graph, 'glob'): graph.glob = torch.tensor( griddata(graph.pos, graph.glob, pos, method=method).astype(np.float32) )
    graph.field  = torch.tensor( griddata(graph.pos, graph.field,  pos, method=method  ).astype(np.float32) )
    graph.target = torch.tensor( griddata(graph.pos, graph.target, pos, method=method  ).astype(np.float32) )
    graph.omega  = torch.tensor( griddata(graph.pos, graph.omega,  pos, method='linear').astype(np.float32) )
    graph.bound  = torch.tensor( np.round(griddata(graph.pos, graph.bound,  pos, method='linear')), dtype=torch.uint8 )
    graph.omega[graph.omega >= 0.9] = 1.
    graph.omega[graph.omega <  0.9] = 0.
    # Update graph.pos
    graph.pos = pos
    return graph


class InterpolateNodes():
    r"""Transformation class to interpolate the fields of a graph to a new set of nodes.
    
    Args:
        pos (torch.Tensor): New node positions.

    Methods:
        __call__(Graph) -> Graph: Interpolates the fields of a graph to a new set of nodes.    
    """

    def __init__(self, pos: torch.Tensor) -> None:
        self.pos = pos

    def __call__(self, graph: Graph) -> Graph:
        # Interpolate
        return interpolate_nodes(graph, self.pos)


class InterpolateNodesToXml():
    r"""Transformation class to interpolate the fields of a graph to a new set of nodes given as the nodes in 
    a NekMesh-generated xml file.
    
    Args:
        xml_file (str): Path to the xml file or a directory containing the xml files.
        num_meshes (Union[int, str], optional): Number of randomly selected meshes to use from a directory.
            If `'all'`, all the meshes in the directory are used. Defaults to `'all'`.

    Methods:
        __call__(Graph) -> Graph: Interpolates the fields of a graph to a new set of nodes given as the nodes in
            a NekMesh-generated xml file.    
    """

    def __init__(self,
                 xml_file: str,
                 num_meshes: Union[int, str] = "all"):
        # Check num_meshes
        if isinstance(num_meshes, str):
            assert num_meshes == "all", "num_meshes must be an integer or 'all'"
        # Get xml files
        if xml_file[-4:] == ".xml": # If xml_file is a single file
            self.xml_files = [xml_file]
        elif xml_file[-4:] == "_xml": # If xml_file is a directory
            self.xml_files = [os.path.join(xml_file, f) for f in sorted(os.listdir(xml_file))]
            if num_meshes == "all": num_meshes = len(self.xml_files)
            self.xml_files = random.choices(self.xml_files, k=num_meshes)

    def __call__(self, graph: Graph) -> Graph:
        # Open and read a random xml file
        dom = ElementTree.parse(random.choice(self.xml_files))
        V = dom.findall('GEOMETRY/VERTEX/V')
        # Get new pos
        dim = graph.pos.size(1)
        pos = torch.tensor([list(map(float, v.text.split()[:dim])) for v in V], dtype=torch.float32)
        # Interpolate to new pos
        return interpolate_nodes(graph, pos)
    

def get_knn_interpolate_weights(pos_x: torch.Tensor,
                            pos_y: torch.Tensor, 
                            k: int,
                            batch_x: Optional[Union[torch.Tensor,None]] = None,
                            batch_y: Optional[Union[torch.Tensor,None]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Computes the indices and weights for the interpolation of the fields of a graph to a new set of nodes using the k-nearest
    neighbors method. These indices and weigths can be used with the `knn_interpolate` function from `graphs`
    
    Args:
        pos_x (torch.Tensor): New node positions.
        pos_y (torch.Tensor): Old node positions.
        k (int): Number of nearest neighbors.
        batch_x (Optional[Union[torch.Tensor,None]], optional): Batch vector for the new nodes. Defaults to `None`.
        batch_y (Optional[Union[torch.Tensor,None]], optional): Batch vector for the old nodes. Defaults to `None`.    
    """
    y_idx, x_idx = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y)
    diff = pos_x[x_idx] - pos_y[y_idx]
    squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
    weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
    return y_idx, x_idx, weights



class BuildKnnInterpWeights():
    r"""Transformation class to compute the indices and weights for the interpolation upsampling in gMuS-GNNs and REMuS-GNNs.
     
    Args:
        k (int): Number of nearest neighbors in $V^1$ for every node in $V^2$.

    Methods:
        __call__(Graph) -> Graph: Computes the indices and weights for the interpolation upsampling in gMuS-GNNs and REMuS-GNNs.
            The classes that need them are the `graphs4cfd.blocks.UpEdgeMP` in REMuS-GNNs and the gMuS-GNNs.
    """

    def __init__(self, k: int):
        self.k = k
        
    def __call__(self, graph: Graph) -> Graph:
        is_batch = hasattr(graph, 'batch') and graph.batch is not None
        if hasattr(graph, 'coarse_mask2'): # For gMuS-GNN with 2 levels
            graph.y_idx_21, graph.x_idx_21, graph.weights_21 = get_knn_interpolate_weights(graph.pos[graph.coarse_mask2], graph.pos, self.k, graph.batch[graph.coarse_mask2] if is_batch else  None, graph.batch if is_batch else None)
            if hasattr(graph, 'coarse_mask3'): # For gMuS-GNN with 3 levels
                graph.y_idx_32, graph.x_idx_32, graph.weights_32 = get_knn_interpolate_weights(graph.pos[graph.coarse_mask3], graph.pos[graph.coarse_mask2], self.k, graph.batch[graph.coarse_mask3] if is_batch else None, graph.batch[graph.coarse_mask2] if is_batch else None)
                if hasattr(graph, 'coarse_mask4'): # For gMuS-GNN with 4 levels
                    graph.y_idx_43, graph.x_idx_43, graph.weights_43 = get_knn_interpolate_weights(graph.pos[graph.coarse_mask4], graph.pos[graph.coarse_mask3], self.k, graph.batch[graph.coarse_mask4] if is_batch else None, graph.batch[graph.coarse_mask3] if is_batch else None)
        return graph