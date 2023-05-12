import torch
import numpy as np
from torch_geometric.nn import knn_graph
from typing import Tuple, Optional, Union

from ..graph import Graph


def connect_knn(pos: torch.Tensor,
                k: int,
                period: Optional[Union[None, Tuple]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Connects nodes using the k-nearest neighbors algorithm.
    
    Args:
        pos (torch.Tensor): Node positions.
        k (int): Number of nearest neighbors.
        period (Optional[Tuple[float, float]], optional): Period of the domain along each axis.
            If None, the domain is not periodic. If an element is `None`, the corresponding axis is not periodic.
            If an element is "auto", the corresponding axis is periodic and the period is computed as the difference
            between the maximum and minimum values of the corresponding axis.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Edge index and edge attributes.
    """
    dim = pos.size(1) # Dimension of the problem
    if dim == 2:
        if period is None:
            dx, dy = None, None
        else:
            dx, dy = period
    elif dim == 3:
        if period is None:
            dx, dy, dz = None, None, None
        else:
            dx, dy, dz = period
    else:
        raise ValueError(f"Invalid dimension: {dim}, must be 2 or 3.")
    # Compute coordinates for knn algorithm
    if dx is not None: # If periodicity is stablished along the x-axis
        if dx == "auto": dx = pos[:,0].max() - pos[:,0].min()
        x = torch.stack( (torch.cos(2*np.pi/dx*pos[:,0]), torch.sin(2*np.pi/dx*pos[:,0])), dim=1 )
    else: # If periodicity is not stablished along the x-axis
        x = pos[:,0].unsqueeze(1)
    if dy is not None: # If periodicity is stablished along the y-axis
        if dy == "auto": dy = pos[:,1].max() - pos[:,1].min()
        y = torch.stack( (torch.cos(2*np.pi/dy*pos[:,1]), torch.sin(2*np.pi/dy*pos[:,1])), dim=1 )
    else: # If periodicity is not stablished along the y-axis
        y = pos[:,1].unsqueeze(1)
    if dim == 3:
        if dz is not None: # If periodicity is stablished along the z-axis
            if dz == "auto": dz = pos[:,2].max() - pos[:,2].min()
            z = torch.stack( (torch.cos(2*np.pi/dz*pos[:,2]), torch.sin(2*np.pi/dz*pos[:,2])), dim=1 )
        else: # If periodicity is not stablished along the z-axis
            z = pos[:,2].unsqueeze(1)
    # Concatenate coordinates
    coordinates = torch.cat( (x, y), dim=1) if dim == 2 else torch.cat( (x, y, z), dim=1)
    # Compute edge_index applying knn to the coordinates
    edge_index = knn_graph(coordinates, k=k)
    # Compute edge_attr
    edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    # Apply periodicity to edge_attr
    if dx is not None:
        edge_attr[edge_attr[:,0] < -dx/2.,0] += dx
        edge_attr[edge_attr[:,0] >  dx/2.,0] -= dx
    if dy is not None:
        edge_attr[edge_attr[:,1] < -dy/2.,1] += dy
        edge_attr[edge_attr[:,1] >  dy/2.,1] -= dy
    if dim == 3 and dz is not None:
        edge_attr[edge_attr[:,2] < -dz/2.,2] += dz
        edge_attr[edge_attr[:,2] >  dz/2.,2] -= dz
    return edge_index, edge_attr


class ConnectKNN():
    """Transformation class to connect nodes using the k-nearest neighbors algorithm.
    
    Args:
        k (int): Number of nearest neighbors.
        period (Optional[Tuple[float, float]], optional): Period of the domain along the $x$- and $y$-axes.

    Methods:
        __call__(graph: Graph) -> Graph: Connects nodes using the k-nearest neighbors algorithm.    
    """

    def __init__(self,
                 k: int,
                 period: Optional[Union[Tuple[float,float], Tuple[float,float,float]]] = (None, None),):
        self.k = k
        self.period = period

    def __call__(self, graph: Graph):
        graph.edge_index, graph.edge_attr = connect_knn(graph.pos, self.k, period=self.period)
        return graph