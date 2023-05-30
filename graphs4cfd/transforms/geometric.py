import torch
import numpy as np
from typing import Union, Optional, Iterable

from ..graph import Graph


def validate_eq(eq: Optional[Union[str,None]] = None,
                format: Optional[Union[str,None]] = None) -> None:
    if eq is not None:
        eq = eq.lower()
        if eq == "ns":
            # If NS equations are used, check that the format is correct
            assert format is not None, "format must be specified for NS equations"
            if format != "uvp" and format != "uv":
                raise ValueError(f"Unknown format {format}, must be 'uvp' or 'uv'")
        elif eq == "adv":
            pass
        else:
            raise ValueError(f"Unknown equation type {eq}, must be 'ns' or 'adv'")
        

def validate_theta(theta: Union[float, Iterable[float]], dim: int) -> None:
    if dim == 2:
        assert isinstance(theta, float), "theta must be a float"
    elif dim == 3:
        assert isinstance(theta, Iterable), "theta must be an iterable"
        assert len(theta) == 3, "theta must be an iterable of length 3"
    else:
        raise ValueError("dim must be 2 or 3")


def rotate_graph(graph: Graph,
           theta: Union[float, Iterable[float]], 
           eq: Optional[Union[str,None]] = None,
           format: Optional[Union[str,None]] = None) -> Graph:
    r"""Rotate the graph. If the graph is 2D, a rotation around the z-axis is performed.
    If the graph is 3D, a Tait-Bryan rotation is performed.
    
    Args:
        graph (Graph): Graph to be rotated.
        theta (float or iterable): Rotation angle(s) in degrees.
        eq (str, optional): Type of equations. It can be `"ns"` for Navier-Stokes equations, `"adv"` for advection equations
        or `None` for no equations. Defaults to `None`.
        format (str, optional): Format of the Navier-Stokes equations. It can be `"uvp"` for velocity and pressure inputs or
        `"uv"` for velocity inputs. Defaults to `None`.

    Returns:
        Graph: Rotated graph.
    """
    
    validate_eq(eq, format)
    # Dimension of the problem
    dim = graph.pos.size(1)
    validate_theta(theta, dim)
    # Convert to radians
    theta = np.deg2rad(theta)

    # Compute rotation matrix
    if dim == 2:
        # 2D rotation matrix
        R = torch.tensor([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], dtype=torch.float32)
    if dim == 3:
        # 3D rotation matrix for Tait-Bryan angles
        R = torch.tensor([[np.cos(theta[0])*np.cos(theta[1]), np.cos(theta[0])*np.sin(theta[1])*np.sin(theta[2])-np.sin(theta[0])*np.cos(theta[2]), np.cos(theta[0])*np.sin(theta[1])*np.cos(theta[2])+np.sin(theta[0])*np.sin(theta[2])],
                          [np.sin(theta[0])*np.cos(theta[1]), np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2])+np.cos(theta[0])*np.cos(theta[2]), np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2])-np.cos(theta[0])*np.sin(theta[2])],
                          [-np.sin(theta[1])                , np.cos(theta[1])*np.sin(theta[2])                                                   , np.cos(theta[1])*np.cos(theta[2])                                                   ]], dtype=torch.float32)
    
    # Rotate position
    graph.pos = (R*graph.pos.unsqueeze(-1)).sum(dim=1)
    # Rotate edge unit vectors
    if hasattr(graph, 'angle_index'):
        '''
        For angle index formulation the edge_attr are invariant to rotation and must be not rotated
        '''
        if hasattr(graph, 'edgeUnitVector'):
            graph.edgeUnitVector = (R*graph.edgeUnitVector.unsqueeze(-1)).sum(dim=1)
            graph.edgeUnitVectorInverse = graph.edgeUnitVector.view(graph.num_nodes, -1, 2).pinverse()
        if hasattr(graph, 'edgeUnitVector2'):
            graph.edgeUnitVector2 = (R*graph.edgeUnitVector2.unsqueeze(-1)).sum(dim=1)
            graph.edgeUnitVectorInverse2 = graph.edgeUnitVector2.view(graph.coarse_mask2.sum().item(), -1, 2).pinverse()
        if hasattr(graph, 'edgeUnitVector3'):
            graph.edgeUnitVector3 = (R*graph.edgeUnitVector3.unsqueeze(-1)).sum(dim=1)
            graph.edgeUnitVectorInverse3 = graph.edgeUnitVector3.view(graph.coarse_mask3.sum().item(), -1, 2).pinverse()
        if hasattr(graph, 'edgeUnitVector4'):
            graph.edgeUnitVectorInverse4 = graph.edgeUnitVector4.view(graph.coarse_mask4.sum().item(), -1, 2).pinverse()
            graph.edgeUnitVector4 = (R*graph.edgeUnitVector4.unsqueeze(-1)).sum(dim=1)
    # Rotate edge attributes
    else:
        if hasattr(graph, 'edge_attr' ) and graph.edge_attr is not None: graph.edge_attr  = (R*graph.edge_attr .unsqueeze(-1)).sum(dim=1)
        if hasattr(graph, 'edge_attr2'): graph.edge_attr2 = (R*graph.edge_attr2.unsqueeze(-1)).sum(dim=1)
        if hasattr(graph, 'edge_attr3'): graph.edge_attr3 = (R*graph.edge_attr3.unsqueeze(-1)).sum(dim=1)
        if hasattr(graph, 'edge_attr4'): graph.edge_attr4 = (R*graph.edge_attr4.unsqueeze(-1)).sum(dim=1)
    # If Adv equation rotate only local fields
    if eq == 'adv':
        graph.loc = (R*graph.loc.unsqueeze(-1)).sum(dim=1)
    # If NS equation rotate u and v
    if eq == 'ns':
        # Rotate only u and v in field
        if format == 'uvp':
            # Rotate input field
            for idx in range(0, graph.field.shape[1],  3):
                graph.field[:,idx:idx+2] = (R*graph.field[:,idx:idx+2].unsqueeze(-1)).sum(dim=1)
            # Rotate target field
            for idx in range(0, graph.target.shape[1], 3):
                graph.target[:,idx:idx+2] = (R*graph.target[:,idx:idx+2].unsqueeze(-1)).sum(dim=1)
        if format == 'uv':
            # Rotate input field
            for idx in range(0, graph.field.shape[1],  2):
                graph.field[:,idx:idx+2] = (R*graph.field[:,idx:idx+2].unsqueeze(-1)).sum(dim=1)
            # Rotate target field
            for idx in range(0, graph.target.shape[1], 2):
                graph.target[:,idx:idx+2] = (R*graph.target[:,idx:idx+2].unsqueeze(-1)).sum(dim=1)
    return graph


class RandomGraphRotation():
    r"""Transformation class for random rotation of a graph.
    
    Args:
        eq (str, optional): Type of equations. It can be `"ns"` for Navier-Stokes equations, `"adv"` for advection equations
        or `None` for no equations. Defaults to `None`.
        format (str, optional): Format of the Navier-Stokes equations. It can be `"uvp"` for velocity and pressure inputs or
        `"uv"` for velocity inputs. Defaults to `None`.

    Methods:
        __call__(graph)->Graph: Randomly rotate the graph.
    """

    def __init__(self,
                 eq: Optional[Union[str,None]] = None,
                 format: Optional[Union[str,None]] = None):
        self.eq = eq
        self.format = format

    def __call__(self, graph: Graph) -> Graph:
        dim = graph.pos.size(1) # Get dimension
        # Random rotation angle(s)
        theta = np.random.uniform(0, 360) if dim == 2 else np.random.uniform(0, 360, size=(3,))
        return rotate_graph(graph, theta, eq=self.eq, format=self.format)


class GraphRotation():
    r"""Transformation class for random rotation of a graph.  If the graph is 2D, a rotation around the z-axis is performed.
    If the graph is 3D, a Tait-Bryan rotation is performed.
    
    Args:
        theta (int, np.ndarray): Rotation angle(s) in degrees.
        eq (str, optional): Type of equations. It can be `"ns"` for Navier-Stokes equations, `"adv"` for advection equations
        or `None` for no equations. Defaults to `None`.
        format (str, optional): Format of the Navier-Stokes equations. It can be `"uvp"` for velocity and pressure inputs or
        `"uv"` for velocity inputs. Defaults to `None`.

    Methods:
        __call__(graph)->Graph: Rotate the graph.
    """

    def __init__(self,
                 theta: Union[int, np.ndarray],
                 eq: Optional[Union[str,None]] = None,
                 format: Optional[Union[str,None]] = None):
        self.theta = theta
        self.eq = eq
        self.format = format

    def __call__(self, graph):
        return rotate_graph(graph, self.theta, eq=self.eq, format=self.format)


def flip_graph_dim(graph: Graph,
                   dim: int,
                   eq: Optional[Union[str,None]] = None,
                   format: Optional[Union[str,None]] = None) -> Graph:
    r"""Flip a graph along a dimension.

    Args:
        graph (Graph): Graph to flip.
        dim (int): Dimension along which to flip the graph.
        eq (str, optional): Type of equations. It can be `"ns"` for Navier-Stokes equations, `"adv"` for advection equations
        or `None` for no equations. Defaults to `None`.
        format (str, optional): Format of the Navier-Stokes equations. It can be `"uvp"` for velocity and pressure inputs or
        `"uv"` for velocity inputs. Defaults to `None`.

    Returns:
        Graph: Flipped graph.
    """

    validate_eq(eq, format)
    max_dim = graph.pos.size(1) # Get dimension
    # Validate flip dimension
    if dim >= max_dim:
        raise ValueError(f"Dimension {dim} is greater than the maximum dimension of the graph ({max_dim})")
    
    # Flip all the vector fields
    graph.pos[:,dim] = -graph.pos[:,dim]
    if hasattr(graph, 'loc'       ): graph.loc[:,dim] = -graph.loc[:,dim]
    if hasattr(graph, 'angle_index'):
        # TODO: Implement flipping graphs with angle_index
        raise ValueError("Flipping graphs with angle_index is not supported")       
    else:
        if hasattr(graph, 'edge_attr' ): graph.edge_attr [:,dim] = -graph.edge_attr [:,dim]
        if hasattr(graph, 'edge_attr2'): graph.edge_attr2[:,dim] = -graph.edge_attr2[:,dim]
        if hasattr(graph, 'edge_attr3'): graph.edge_attr3[:,dim] = -graph.edge_attr3[:,dim]
        if hasattr(graph, 'edge_attr4'): graph.edge_attr4[:,dim] = -graph.edge_attr4[:,dim]
    eq = eq.lower()
    if eq == 'adv':
        pass
    elif eq == 'ns':
        if format == 'uvp':
            graph.field [:,dim::3] = -graph.field [:,dim::3]
            graph.target[:,dim::3] = -graph.target[:,dim::3]
        else:
            graph.field [:,dim::2] = -graph.field [:,dim::2]
            graph.target[:,dim::2] = -graph.target[:,dim::2]
    return graph



class RandomGraphFlip():
    r"""Transformation class for random flipping of a graph.
    
    Args:
        x_flip (bool, optional): If `True`, the graph is flipped along the x-axis with a 50\% probability.
            Defaults to `True`.
        y_flip (bool, optional): If `True`, the graph is flipped along the y-axis with a 50\% probability.
            Defaults to `True`.
        z_flip (bool, optional): If `True`, the graph is flipped along the z-axis with a 50\% probability.
            Defaults to `True`.
        eq (str, optional): Type of equations. It can be `"ns"` for Navier-Stokes equations, `"adv"` for advection equations
        or `None` for no equations. Defaults to `None`.
        format (str, optional): Format of the Navier-Stokes equations. It can be `"uvp"` for velocity and pressure inputs or
        `"uv"` for velocity inputs. Defaults to `None`.

    Methods:
        __call__(graph)->Graph: Randomly flip the graph.
    """

    def __init__(self,
                 x_flip: Optional[bool] = True,
                 y_flip: Optional[bool] = True,
                 z_flip: Optional[bool] = True,
                 eq: Optional[Union[str,None]] = None,
                 format: Optional[Union[str,None]] = None):
        self.flip = (x_flip, y_flip, z_flip)
        self.eq = eq
        self.format = format

    def __call__(self, graph):
        dim = graph.pos.size(1)
        for axis, flag in enumerate(self.flip[:dim]):
            if flag and np.random.randint(2): 
                graph = flip_graph_dim(graph, axis, eq=self.eq, format=self.format)
        return graph