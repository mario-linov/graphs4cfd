import torch
from torch import nn
from typing import Tuple, Union, Optional, Callable
from torch_geometric.utils import remove_self_loops, scatter, coalesce

from ..graph import Graph


def restriction(graph: Graph,
                coarse_mask: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                num_nodes: int,
                device: torch.device) -> None:
    r"""Restricts a graph to a subset of nodes. To be used in gMuS-GNN models.

    Args:
        graph (Graph): The graph to restrict.
        coarse_mask (torch.Tensor): A mask indicating which nodes are kept in the graph.
        edge_attr (torch.Tensor): The edge attributes of the graph.
        edge_index (torch.Tensor): The edge indices of the graph.
        num_nodes (int): The number of nodes in the high-resolution graph.
        device (torch.device): The device to use.

    Returns:
        None: The graph is modified in-place.
    """
    mask2idx = -torch.ones(num_nodes, dtype=torch.long, device=device)
    mask2idx[coarse_mask] = torch.arange(graph.field.size(0), dtype=torch.long, device=device)
    graph.edge_index = mask2idx[edge_index]
    graph.edge_attr = edge_attr
    return

def knn_interpolate(x: torch.Tensor, y_idx: torch.Tensor, x_idx: torch.Tensor, weights: torch.Tensor):
    r"""Interpolates the fields of a graph to a new set of nodes using the k-nearest neighbors method.
    The arguments `y_idx`, `x_idx` and `weights` can be obtained with the `graphs4cfd.transforms.interpolate.get_knn_interpolate_weights` function
    or by applying the transformation class `graphs4cfd.transforms.interpolate.KnnInterpolateWeights` to a graph.

    Args:
        x (torch.Tensor): The fields of the graph to interpolate.
        y_idx (torch.Tensor): The interpolation indices of the new nodes.
        x_idx (torch.Tensor): The interpolation indices of the old nodes.
        weights (torch.Tensor): The weights of the interpolation.
    """
    y_num_nodes = y_idx.max().item()+1
    y  = scatter(x[x_idx]*weights, y_idx, dim=0, dim_size=y_num_nodes, reduce='sum')
    y /= scatter(weights, y_idx, dim=0, dim_size=y_num_nodes, reduce='sum')
    return y


def pool_edge(idxHR_to_idxLR: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, aggr: str="mean"):
    r"""Pools the edges of a graph to a new set of edges using the idxHR_to_idxLR mapping.

    Args:
        idxHR_to_idxLR (torch.Tensor): A mapping from the old node (or higher resolution) indices to the new (or lower resolution) node indices.
        edge_index (torch.Tensor): The old edge indices.
        edge_attr (torch.Tensor): The old edge attributes.
        aggr (str, optional): The aggregation method. Can be "mean" or "sum". Defaults to "mean".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The new edge indices and attributes.
    """
    num_nodes = idxHR_to_idxLR.max().item()+1 # number of nodes in the lower resolution graph
    edge_index = idxHR_to_idxLR[edge_index.view(-1)].view(2, -1) # edge indices in the lower resolution graph
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr) # remove self loops
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, reduce=aggr) # aggregate edges
    return edge_index, edge_attr


def lstsq(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solves the least squares problem $AX=B$ for $X$.

    Args:
        A (torch.Tensor): The matrix $A$.
        B (torch.Tensor): The matrix $B$.

    Returns:
        torch.Tensor: The matrix $X$.    
    """
    if int(torch.__version__.split('.')[1]) < 9:
        device = A.device
        return torch.bmm(torch.pinverse(A.to("cpu")).to(device), B)
    else:
        return A.pinverse() @ B


def edgeScalarToNodeVector(edge_attr: torch.Tensor,
                           edge_index: torch.Tensor,
                           edgeUnitVector: Optional[Union[None, torch.Tensor]] = None,
                           edgeUnitVectorInverse: Optional[Union[None, torch.Tensor]] = None,
                           coarse_mask: Optional[Union[None, torch.Tensor]] = None) -> torch.Tensor:
    r""" For each node $j$, find the least square solution to $[e_{ij}][u_{j}] = [u_{ij}]$. Used for the projection-aggregation step in REMuS-GNNs.
    Matrix $[e_{ij}]$ contains unitary edge vectors in the rows [kx2], $[u_j]$ is the solution vector at node j [2xFe]
    and $[u_{ij}]$ is the projection of $u_j$ along each edge $(i,j)$ [kxFe], where Fe is the number of edge features.

    Args:
        edge_attr (torch.Tensor): Features of every edge. Dimensions [|E|xFe].
        edge_index (torch.Tensor): The edge indices. Dimensions [2x|E|].
        edgeUnitVector (torch.Tensor, optional): Unitary vectors of every edge. Dimensions [|E|x2].
        edgeUnitVectorInverse (torch.Tensor, optional): Inverse of the unitary vectors of every edge. Dimensions [|E|x2].
        coarse_mask (torch.Tensor, optional): A mask indicating which nodes are kept in the graph. Only used for low resolution graphs. Dimensions [|V|].

    Returns:
        torch.Tensor: The solution vector at every node. Dimensions [|V|x2Fe].
    """
    assert (edgeUnitVector is None) != (edgeUnitVectorInverse is None), "Either edgeUnitVector or edgeUnitVectorInverse must be provided."
    num_features = edge_attr.size(1)
    num_nodes = edge_index.max().item()+1 if coarse_mask is None else coarse_mask.sum().item()
    if edgeUnitVectorInverse is not None:
        v = edgeUnitVectorInverse @ edge_attr.view(num_nodes, -1, num_features)
    else:
        v = lstsq(edgeUnitVector.view(num_nodes, -1, 2), edge_attr.view(num_nodes, -1, num_features))
    return v.transpose(dim0=1, dim1=2).flatten(start_dim=1, end_dim=2)


class MLP(nn.Module):
    r"""A multi-layer perceptron (MLP) with SELU activation functions.
    
    Args:
        input_size (int): The size of the input.
        layers_width (List[int]): The width of each layer, excluding the input layer.
        layer_norm (bool, optional): Whether to apply layer normalization after the last layer. Defaults to False.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Computes the forward pass of the MLP.
    """

    def __init__(self,
                 input_size: int,
                 layers_width: Tuple[int],
                 layer_norm: bool = False):
        super().__init__()
        self.MLP = nn.Sequential()
        self.MLP.add_module("linear_1", nn.Linear(input_size, layers_width[0]))
        self.MLP.add_module("selu_1",   nn.SELU())
        for i in range(len(layers_width)-2):
            self.MLP.add_module("linear_"+str(i+2), nn.Linear(layers_width[i], layers_width[i+1]))
            self.MLP.add_module("selu_"+str(i+2),   nn.SELU())
        self.MLP.add_module("linear_"+str(len(layers_width)), nn.Linear(layers_width[-2], layers_width[-1]))
        if layer_norm: self.MLP.add_module("layer_norm", nn.LayerNorm(layers_width[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MLP(x)


class GNBlock(nn.Module):
    r"""A graph neural network block based on Battaglia et al. (2018) (https://arxiv.org/abs/1806.01261).
    
    Args:
        edge_mlp_args (Tuple): Arguments for the MLP used for updating the edge features.
        node_mlp_args (Tuple): Arguments for the MLP used for updating the node features.
        aggr (str, optional): The aggregation operator to use. Can be 'mean' or 'sum'. Defaults to 'mean'.

    Methods:
        reset_parameters(): Reinitializes the parameters of the MLPs.
        forward(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor: Computes the forward pass of the GN block.
    """

    def __init__(self,
                 edge_mlp_args: Tuple,
                 node_mlp_args: Tuple,
                 aggr: str = 'mean'):
        super().__init__()
        self.edge_mlp = MLP(*edge_mlp_args)
        self.node_mlp = MLP(*node_mlp_args)
        self.aggr = aggr

    def reset_parameters(self):
        models = [model for model in [self.node_mlp, self.edge_mlp] if model is not None]
        for model in models:
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()

    def forward(self,
                v: torch.Tensor,
                e: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        # Edge update
        e = self.edge_mlp( torch.cat((e, v[row], v[col]), dim=-1) )
        # Edge aggregation
        aggr = scatter(e, col, dim=0, dim_size=v.size(0), reduce=self.aggr)
        # Node update
        v = self.node_mlp( torch.cat((aggr, v), dim=-1) )
        return v, e


# Alias for GNBlock
MP = GNBlock


class DownMP(nn.Module):
    r"""DownMP from Lino et al. (2022) (https://doi.org/10.1063/5.0097679)
    
    Args:
        down_mlp_args (Tuple): Arguments for the MLP used for the downsampling edge-model.
        hr_graph_idx (int): The index of the high-resolution graph.

    Methods:
        reset_parameters(): Reinitializes the parameters of the MLPs.
        forward(graph: Graph, activation: Optional[Callable] = None) -> Graph: Computes the forward pass of the DownMP.
    """

    def __init__(self,
                 down_mlp_args: Tuple,
                 hr_graph_idx: int):
        super().__init__()
        self.down_mlp = MLP(*down_mlp_args) # The MLP used for the downsampling edge-model
        self.hr_graph_idx = hr_graph_idx # The index of the high-resolution graph
        self.lr_graph_idx = hr_graph_idx + 1 # The index of the low-resolution graph
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.down_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self,
                graph: Graph, 
                activation: Optional[Callable] = None) -> Graph:
        # Get needed variables
        graph.pos       = getattr(graph, f'pos_{self.lr_graph_idx}')
        cluster         = getattr(graph, f'cluster_{self.lr_graph_idx}')
        mask            = getattr(graph, f'mask_{self.lr_graph_idx}')
        idxHr_to_idxLr  = getattr(graph, f'idx{self.hr_graph_idx}_to_idx{self.lr_graph_idx}')
        e               = getattr(graph, f'e_{self.hr_graph_idx}{self.lr_graph_idx}')
        # Appy edge-model
        e = self.down_mlp( torch.cat((e, graph.field), dim=-1) )
        # Aggregate the edge features
        graph.field = scatter(e, cluster, dim=0, reduce='mean')[mask]
        # Apply activation
        if activation is not None:
                graph.field = activation(graph.field)
        # Aggregate the edges
        graph.edge_index, graph.edge_attr = pool_edge(idxHr_to_idxLr, graph.edge_index, graph.edge_attr, aggr="mean")
        return graph


class UpMP(nn.Module):
    r"""UpMP from Lino et al. (2022) (https://doi.org/10.1063/5.0097679)
    
    Args:
        up_mlp_args (Tuple): Arguments for the MLP used for the upsampling edge-model.
        lr_graph_idx (int): The index of the low-resolution graph.
        
    Methods:
        reset_parameters(): Reinitializes the parameters of the MLPs.
        forward(graph: Graph, field_hr_old: torch.Tensor, pos_hr: torch.Tensor, activation: Optional[Callable] = None) -> Graph: Computes the forward pass of the UpMP.
    """
    def __init__(self,
                 up_mlp_args: Tuple,
                 lr_graph_idx: int):
        super().__init__()
        self.up_mlp = MLP(*up_mlp_args)
        self.lr_graph_idx = lr_graph_idx
        self.hr_graph_idx = lr_graph_idx - 1
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.up_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self,
                graph: Graph,
                field_hr_old: torch.Tensor,
                pos_hr: torch.Tensor,
                activation: Optional[Callable] = None) -> Graph:
        r"""Computes the forward pass of the UpMP.
        
        Args:
            graph (Graph): The graph object.
            field_hr_old (torch.Tensor): The previous field of the high-resolution graph (skip connection).
            pos_hr (torch.Tensor): The positions of the high-resolution graph.
            activation (Optional[Callable]): The activation function to be used.

        Returns:
            Graph: The graph object.
        """
        idxHr_to_idxLr = getattr(graph, f'idx{self.hr_graph_idx}_to_idx{self.lr_graph_idx}')
        # Realtive position 
        e = -getattr(graph, f'e_{self.hr_graph_idx}{self.lr_graph_idx}')
        # Edge-model
        graph.field = self.up_mlp( torch.cat((e, graph.field[idxHr_to_idxLr], field_hr_old), dim=-1) )
        # Restore dense pos and batch
        graph.pos = pos_hr
        if activation is not None:
            graph.field = activation(graph.field)
        return graph


class EdgeMP(nn.Module):
    r"""EdgeMP from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).
    
    Args:
        angle_mlp_args (Tuple): Arguments for the MLP used for the angle-model.
        edge_mlp_args (Tuple): Arguments for the MLP used for the edge-model.
        aggr (str): The aggregation operator to be used. Either "mean" or "sum".

    Methods:
        reset_parameters(): Reinitializes the parameters of the MLPs.
        forward(e: torch.Tensor, a: torch.Tensor, angle_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: Computes the forward pass of the EdgeMP.
            Updated edge and angle features.
    """

    def __init__(self,
                 angle_mlp_args: Tuple,
                 edge_mlp_args: Tuple,
                 aggr: str = "mean"):
        super().__init__()
        self.angle_mlp = MLP(*angle_mlp_args)
        self.edge_mlp  = MLP(*edge_mlp_args)
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.edge_mlp, self.angle_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self,
                e: torch.Tensor,
                a: torch.Tensor,
                angle_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = angle_index
        # Angle update
        a = self.angle_mlp( torch.cat((a, e[row], e[col]), dim=1) )
        # Aggregation
        aggr = scatter(a, col, dim=0, dim_size=e.size(0), reduce=self.aggr)
        # Edge update
        e = self.edge_mlp( torch.cat((aggr, e), dim=1) )
        return e, a


class DownEdgeMP(nn.Module):
    r"""DownEdgeMP from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    Args:
        angle_mlp_args (Tuple): Arguments for the MLP (`graphs4cfd.blocks.MLP`) used for the angle-model.
        edge_mlp_args (Tuple): Arguments for the MLP (`graphs4cfd.blocks.MLP`) used for the edge-model.

    Methods:
        reset_parameters(): Reinitializes the parameters of the MLPs.
        forward(e1: torch.Tensor, e2: torch.Tensor, a12: torch.Tensor, angle_index12: torch.Tensor) -> torch.Tensor: Computes the forward pass of the DownEdgeMP.
            Return the updated edge-features of the low-resolution graph.
    """

    def __init__(self, angle_mlp_args: Tuple, edge_mlp_args: Tuple):
        super().__init__()
        self.angle_mlp = MLP(*angle_mlp_args)
        self.edge_mlp  = MLP(*edge_mlp_args)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.edge_mlp, self.angle_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self,
                e1: torch.Tensor,
                e2: torch.Tensor,
                a12: torch.Tensor,
                angle_index12: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            e1 (torch.Tensor): The edge features of the high-resolution graph. Dimension: $|E^1| \times F^1_e$.
            e2 (torch.Tensor): The edge features of the low-resolution graph. Dimension: $|E^2| \times F^2_e$.
            a12 (torch.Tensor): The features of the angles directed from an edges in $E^1$ to an edge in $E^2$. Dimension: $|E^2|k_1 \times F^{12}_a$. 
            angle_index12 (torch.Tensor): The indices of the angles directed from an edges in $E^1$ to an edge in $E^2$.
                The indices of the sender edges, `angle_index12[0]`, are expressed using the $E^1$ edge indices and the indices of the receiver edge
                are expressed using the $E^2$ edge indices, `angle_index12[1]`. Dimension: $2 \times |E^2|k_1$.
        """
        row, col = angle_index12
        # Angle update
        a12 = self.angle_mlp( torch.cat([a12, e1[row], e2[col]], dim=1) )
        # Aggregation
        aggr = scatter(a12, col, dim=0, dim_size=e2.size(0), reduce="mean")
        # Edge update
        e2 = self.edge_mlp( torch.cat((aggr, e2), dim=1) )
        return e2


class UpEdgeMP(nn.Module):
    r"""UpEdgeMP from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    Args:
        up_mlp_args (Tuple): Arguments for the MLP (`graphs4cfd.blocks.MLP`) used for the up-model.

    Methods:
        reset_parameters(): Reinitializes the parameters of the MLPs.
        forward(pos: torch.Tensor, y_idx_21: torch.Tensor, x_idx_21: torch.Tensor, weights_21: torch.Tensor, edge_attr2: torch.Tensor,
            edge_index2: torch.Tensor, edgeUnitVectorInverse2: torch.Tensor, coarse_mask2: torch.Tensor, edge_attr1: torch.Tensor,
            edge_index1: torch.Tensor, edgeUnitVector1: torch.Tensor, coarse_mask1=None) -> torch.Tensor: Computes the forward pass of the UpEdgeMP.
            Return the updated edge-features of the high-resolution graph.
    """

    def __init__(self, up_mlp_args: Tuple):
        super().__init__()
        self.up_mlp  = MLP(*up_mlp_args)
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.up_mlp]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self,
                pos: torch.Tensor,
                y_idx_21: torch.Tensor,
                x_idx_21: torch.Tensor,
                weights_21: torch.Tensor,
                edge_attr2: torch.Tensor,
                edge_index2: torch.Tensor,
                edgeUnitVectorInverse2: torch.Tensor,
                coarse_mask2: torch.Tensor,
                edge_attr1: torch.Tensor,
                edge_index1: torch.Tensor,
                edgeUnitVector1: torch.Tensor,
                coarse_mask1=None) -> torch.Tensor:
        r"""
        Args:
            pos (torch.Tensor): The node positions of the low-resolution graph. Dimension: $|V^2| \times 2$.
            y_idx_21 (torch.Tensor): The new graph interpolation indices. The k nearest neighbors in the low-resolution graph for the nodes in the high-resolution graph. Dimension: $|V^1| \times k$.
                Dimension: $|V^1| \times k_1$.
            x_idx_21 (torch.Tensor): The old graph interpolation indices. The nodes in the high-resolution graph that have their neighbour in ´y_idx_21`. Dimension: $|V^1| \times k_1$.
            weights_21 (torch.Tensor): The interpolation weights.
            edge_attr2 (torch.Tensor): The edge features of the low-resolution graph. Dimension: $|E^2| \times F^2_e$.
            edge_index2 (torch.Tensor): The edge indices of the low-resolution graph. Dimension: $2 \times |E^2|$.
            edgeUnitVectorInverse2 (torch.Tensor): Inverse of matrix with the edge unit vectors of the incoming edges at each node in the low-resolution graph.
                Dimension: $|V^2| \times 2 \times k_2$.
            coarse_mask2 (torch.Tensor): The coarse mask of the low-resolution graph. Dimension: $|V^2|$.
            edge_attr1 (torch.Tensor): The edge features of the high-resolution graph. Dimension: $|E^1| \times F^1_e$.
            edge_index1 (torch.Tensor): The edge indices of the high-resolution graph. Dimension: $2 \times |E^1|$.
            edgeUnitVector1 (torch.Tensor): Matrix with the edge unit vectors of the incoming edges at each node in the high-resolution graph. Dimension: $|V^1| \times k_1 \times 2$.
            coarse_mask1 (torch.Tensor): The coarse mask of the high-resolution graph. Dimension: $|V^1|$. Only needed if $V^1$ is not the highest-resolution graph. Default: None.
        
        Returns:
            torch.Tensor: The updated edge features of the high-resolution graph. Dimension: $|E^1| \times F^1_e$.
        """
        total_num_nodes = pos.size(0)
        num_features = edge_attr2.size(1)
        v1 = torch.zeros(total_num_nodes, 2*num_features, device=pos.device)
        '''
        Since edge_index1 is in indeces from level 1 I need v1 to have total_num_nodes size
        '''
        if coarse_mask1 is None: coarse_mask1 = torch.ones(total_num_nodes, dtype=torch.bool, device=pos.device)
        v2 = edgeScalarToNodeVector(edge_attr2, edge_index2, edgeUnitVectorInverse=edgeUnitVectorInverse2, coarse_mask=coarse_mask2) # [|V_2|, 2F]
        # 2-Interpolate to nodes in level 1
        # v1[coarse_mask1] = knn_interpolate(v2, pos[coarse_mask2], pos[coarse_mask1], batch[coarse_mask2], batch[coarse_mask1], k=3) # [|V_1|, 2F]
        v1[coarse_mask1] = knn_interpolate(v2, y_idx_21, x_idx_21, weights_21)
        # 3-Project vectors in the nodes of V_1 [|V_1|, 2F] to scalars in e1 [E_1, F]
        col1 = edge_index1[1]
        e1 = (v1[col1].view(col1.size(0), -1, 2)*edgeUnitVector1.unsqueeze(1)).sum(dim=-1)
        # 4-Cocatenate skip connection from level 1 and reduce feautures size with per edge MLP
        return self.up_mlp( torch.cat([e1, edge_attr1], dim=1) )