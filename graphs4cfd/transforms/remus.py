import torch
from typing import Tuple, Union, Optional

from ..graph import Graph
from .connect import connect_knn
from .mus import guillard_coarsening


def extend_graph(edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r""""Computes the unit vector of each edge, the angle index and the angle attributes for use in REMuS-GNNs.
    It is assumed that all nodes have the same indegree.
    
    Args:
        edge_index (torch.Tensor): Edge index.
        edge_attr (torch.Tensor): Edge attributes.
        k (int): Number of nearest neighbors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: with shapes:
            - edgeUnitVector: |E|x2
            - angle_index: 2xk|E|
            - angle_attr: |A|x4
    """
    k = (edge_index[1]==0).sum().item() # Indegree of the graph. It is assumed that all nodes have the same indegree
    num_edges = edge_index.size(1)
    row, col = edge_index[0], edge_index[1]
    cross_product = lambda a, b: a[:,0]*b[:,1] - a[:,1]*b[:,0]
    # For each edge create a unit vector from sender to receiver node
    edgeSize = edge_attr.norm(2, dim=1, keepdim=True)
    edgeUnitVector = edge_attr/edgeSize
    # Build angles
    angle_index_row = torch.cat([(col==r).nonzero().squeeze(1) for r in row], dim=0)
    angle_index_col = torch.cat([torch.tensor(k*[i]) for i in torch.arange(num_edges)], dim=0)
    angle_index = torch.stack([angle_index_row, angle_index_col], dim=0)
    # Find cos and sin of relative angle between edges
    cosRelAngle = (edgeUnitVector[angle_index_row]*edgeUnitVector[angle_index_col]).sum(dim=1)
    sinRelAngle = cross_product(edgeUnitVector[angle_index_row], edgeUnitVector[angle_index_col])
    # Create angle input features
    angle_attr = torch.cat([edgeSize[angle_index_row], edgeSize[angle_index_col], cosRelAngle.unsqueeze(1), sinRelAngle.unsqueeze(1)], dim=1)
    return edgeUnitVector, angle_index, angle_attr


class ExtendGraph():
    r"""Computes the unit vector of each edge, the angle index and the angle attributes for use in REMuS-GNNs.

    Methods:
        __call__(graph: Graph) -> Graph: Computes the unit vector of each edge, the angle index and the angle attributes for use in REMuS-GNNs.
    
    """
    def __init__(self):
        pass
    
    def __call__(self, graph: Graph) -> Graph:
        graph.edgeUnitVector, graph.angle_index, graph.angle_attr = extend_graph(graph.edge_index, graph.edge_attr)
        graph.edgeUnitVectorInverse = graph.edgeUnitVector.view(graph.num_nodes, -1, 2).pinverse()
        return graph


class BuildRemusGraph():
    r"""Builds a REMuS graph with the specified number of levels.

    Args:
        num_levels (int): Number of levels.
        k (int): Number of nearest neighbors.
        period (Union[None, Tuple], optional): Period of the domain along each axis.
            If None, the domain is not periodic. If an element is `None`, the corresponding axis is not periodic.
            If an element is "auto", the corresponding axis is periodic and the period is computed as the difference
            between the maximum and minimum values of the corresponding axis.
            Defaults to None.
        scale_edge_length (Union[None, Tuple], optional): Scaling of the edge lengths at each level.
            If None, the edge length is not scaled. If an element is `None`, the corresponding axis is not scaled.
            If an element is "auto", the corresponding axis is scaled and the scaling factor is computed as the
            difference between the maximum and minimum values of the corresponding axis.
            Defaults to None.

    Methods:
        __call__(graph: Graph) -> Graph: Builds a REMuS graph with the specified number of levels.
    """
    def __init__(self,
                 num_levels: int,
                 k: int,
                 period: Optional[Union[None, Tuple]] = None,
                 scale_edge_length: Optional[Union[None, Tuple]] = None):
        self.num_levels = num_levels
        self.k = k
        self.period = period
        self.scale_edge_length = scale_edge_length

    def __call__(self, graph: Graph) -> Graph:
        # Connect level 1
        graph.edge_index, graph.edge_attr = connect_knn(graph.pos, self.k, period=self.period)
        if self.scale_edge_length[0] is not None: graph.edge_attr /= (2*self.scale_edge_length[0])
        # Coarsen level 1
        graph.coarse_mask2 = guillard_coarsening(graph.edge_index, graph.num_nodes) # Mask applied to V^1 to obtain V^2
        coarse_index2 = graph.coarse_mask2.nonzero().squeeze() # V^1-index of the nodes in V^2
        # Connect level 2
        graph.edge_index2, graph.edge_attr2 = connect_knn(graph.pos[coarse_index2], self.k, period=self.period)
        if self.scale_edge_length[1] is not None: graph.edge_attr2 /= (2*self.scale_edge_length[1])
        if self.num_levels > 2:
            # Coarsen level 2
            graph.coarse_mask3 = torch.zeros_like(graph.coarse_mask2, dtype=torch.bool) 
            graph.coarse_mask3[graph.coarse_mask2] = guillard_coarsening(graph.edge_index2, graph.coarse_mask2.sum()) # Mask applied to V^1 to obtain V^3
            coarse_index3 = graph.coarse_mask3.nonzero().squeeze() # V^1-index of the nodes in V^3
            # Connect level 3
            graph.edge_index3, graph.edge_attr3 = connect_knn(graph.pos[coarse_index3], self.k, period=self.period)
            if self.scale_edge_length[2] is not None: graph.edge_attr3 /= (2*self.scale_edge_length[2])
            if self.num_levels > 3:
                # Coarsen level 3
                graph.coarse_mask4 = torch.zeros_like(graph.coarse_mask3, dtype=torch.bool)
                graph.coarse_mask4[graph.coarse_mask3] = guillard_coarsening(graph.edge_index3, graph.coarse_mask3.sum()) # Mask applied to V^1 to obtain V^4
                coarse_index4 = graph.coarse_mask4.nonzero().squeeze() # V^1-index of the nodes in V^4
                # Connect level 4
                graph.edge_index4, graph.edge_attr4 = connect_knn(graph.pos[coarse_index4], self.k, period=self.period)
                if self.scale_edge_length[3] is not None: graph.edge_attr4 /= (2*self.scale_edge_length[3])
        # Renumber edge index to the original indices
        graph.edge_index2 = coarse_index2[graph.edge_index2]
        if self.num_levels > 2: graph.edge_index3 = coarse_index3[graph.edge_index3]
        if self.num_levels > 3: graph.edge_index4 = coarse_index4[graph.edge_index4]
        # Extend every graph
        # For G1
        graph.edgeUnitVector, graph.angle_index, graph.angle_attr = extend_graph(graph.edge_index, graph.edge_attr)
        graph.edgeUnitVectorInverse = graph.edgeUnitVector.view(graph.num_nodes, -1, 2).pinverse() # Create edgeUnitVectorInverse required for contraction/des-project
        # For G2
        graph.edgeUnitVector2, graph.angle_index2, graph.angle_attr2 = extend_graph(graph.edge_index2, graph.edge_attr2)
        graph.edgeUnitVectorInverse2 = graph.edgeUnitVector2.view(graph.coarse_mask2.sum().item(), -1, 2).pinverse() # Create edgeUnitVectorInverse required for contraction/des-project
        # For G3
        if self.num_levels > 2:
            graph.edgeUnitVector3, graph.angle_index3, graph.angle_attr3 = extend_graph(graph.edge_index3, graph.edge_attr3)
            graph.edgeUnitVectorInverse3 = graph.edgeUnitVector3.view(graph.coarse_mask3.sum().item(), -1, 2).pinverse() # Create edgeUnitVectorInverse required for contraction/des-project
        # For G4
        if self.num_levels > 3:
            graph.edgeUnitVector4, graph.angle_index4, graph.angle_attr4 = extend_graph(graph.edge_index4, graph.edge_attr4)  
            graph.edgeUnitVectorInverse4 = graph.edgeUnitVector4.view(graph.coarse_mask4.sum().item(), -1, 2).pinverse() # Create edgeUnitVectorInverse required for contraction/des-project
        # Need to define the inter graph angles too ...
        # ... for 1->2 and 2->1
        graph.angle_index12, graph.angle_attr12 = self.angleIndexDownMP(graph.edge_index , graph.edge_attr , graph.edge_index2, graph.edge_attr2, coarse_index2, self.k)
        # ... for 2->3 and 3->2
        if self.num_levels > 2:
            graph.angle_index23, graph.angle_attr23 = self.angleIndexDownMP(graph.edge_index2, graph.edge_attr2, graph.edge_index3, graph.edge_attr3, coarse_index3, self.k)
        # ... for 3->4
        if self.num_levels > 3:
            graph.angle_index34, graph.angle_attr34 = self.angleIndexDownMP(graph.edge_index3, graph.edge_attr3, graph.edge_index4, graph.edge_attr4, coarse_index4, self.k)
        return graph


    @staticmethod
    def angleIndexDownMP(edge_index1, edge_attr1, edge_index2, edge_attr2, coarse_index2, k):
        '''
        Create angle_index12 [2, sum_i k*num_outgoing_edges_i for i in V_2]
        row has the index of edges in E_1 and col the index of edges in E_2
        [(i,j) 1, (i,j) 2, ..., (i,j) 6, ..., (i,j) 1, (i,j) 2, ...]
        [(j,k)'1, (j,k)'1, ..., (j,k)'1, ..., (j,k)'8, (j,k)'8, ...]
        '''
        # Given node j find all i (k nodes) sending to j in level 1 
        in_edges_index1  = torch.cat([(edge_index1[1]==idx).nonzero().flatten() for idx in coarse_index2]) # (i,j)
        # Given node j find all k (not necessarily equal to k nodes) receiving from j in level 2   
        out_edges_index2 =           [(edge_index2[0]==idx).nonzero().flatten() for idx in coarse_index2]
        num_out_edges = torch.tensor([o.numel() for o in out_edges_index2]) # Number of outgoing edges from each k node in V_2
        out_edges_index2 = torch.cat(out_edges_index2, dim=0) # (j, k)'
        # Build row and col of angle_index12
        row = torch.repeat_interleave(in_edges_index1 .view(-1, k), num_out_edges, dim=0).flatten()
        col = torch.repeat_interleave(out_edges_index2            ,             k, dim=0)
        # Create angle attributes
        cross_product = lambda a, b: a[:,0]*b[:,1] - a[:,1]*b[:,0]
        edgeSize1 = edge_attr1.norm(2, dim=1, keepdim=True)
        edgeSize2 = edge_attr2.norm(2, dim=1, keepdim=True)
        edgeUnitVector1 = edge_attr1/edgeSize1
        edgeUnitVector2 = edge_attr2/edgeSize2
        cosRelAngle = (edgeUnitVector1[row]*edgeUnitVector2[col]).sum(dim=1)
        sinRelAngle = cross_product(edgeUnitVector1[row], edgeUnitVector2[col])
        angle_attr12 = torch.cat([edgeSize1[row], edgeSize2[col], cosRelAngle.unsqueeze(1), sinRelAngle.unsqueeze(1)], dim=1)
        return torch.stack([row,col], dim=0), angle_attr12