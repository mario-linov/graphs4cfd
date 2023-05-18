import torch
import torch.nn.functional as F
from typing import Optional

from .model import Model
from .blocks import MLP, EdgeMP, DownEdgeMP, UpEdgeMP, edgeScalarToNodeVector
from ..graph import Graph


class NsRotEquiTreeScaleGNN(Model):
    """The three-scale REMuS-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Angle-functions ################## Edge-functions ##############
        # Encoder
        "angle_encoder"  : (4, (128,128), True),
        "angle_encoder12": (4, (128,128), True),
        "angle_encoder2" : (4, (128,128), True),
        "angle_encoder23": (4, (128,128), True),
        "angle_encoder3" : (4, (128,128), True),
        "edge_encoder"   : (3, (128,128), True),
        "edge_encoder2"  : (3, (128,128), True),
        "edge_encoder3"  : (3, (128,128), True),
        # Level 1
        "mp111":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp112":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp113":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp114":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        # Pooling 1->2
        "down_mp12":   ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        # Level 2
        "mp211":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp212":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        # Pooling 2->3
        "down_mp23":   ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        # Level 3
        "mp31":     ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp32":     ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp33":     ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp34":     ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        # Undown_mping 3->2
        "up_mp32": ((128+128,   (128,128,128), True),), ###### WRONG
        # Level 2
        "mp221":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp222":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        # Undown_mping 2->1
        "up_mp21": ((128+128,   (128,128,128), True),), ###### WRONG
        # Level 1
        "mp121":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp122":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp123":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        "mp124":    ((128+2*128, (128,128), True), (128+128, (128,128), True)),
        # Decoder
        "decoder": (128, (128,1), False),
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_fields = 2

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.angle_encoder   = MLP(*arch["angle_encoder"  ])
        self.angle_encoder12 = MLP(*arch["angle_encoder12"])
        self.angle_encoder2  = MLP(*arch["angle_encoder2" ])
        self.angle_encoder23 = MLP(*arch["angle_encoder23"])
        self.angle_encoder3  = MLP(*arch["angle_encoder3" ])
        self.edge_encoder    = MLP(*arch["edge_encoder"   ])
        self.edge_encoder2   = MLP(*arch["edge_encoder2"  ])
        self.edge_encoder3   = MLP(*arch["edge_encoder3"  ])
        # Level 1
        self.mp111 = EdgeMP(*arch["mp111"])
        self.mp112 = EdgeMP(*arch["mp112"])
        self.mp113 = EdgeMP(*arch["mp113"])
        self.mp114 = EdgeMP(*arch["mp114"])
        # Pooling 1->2
        self.down_mp12 = DownEdgeMP(*arch["down_mp12"])
        # Level 2
        self.mp211 = EdgeMP(*arch["mp211"])
        self.mp212 = EdgeMP(*arch["mp212"])
        # Pooling 2->3
        self.down_mp23 = DownEdgeMP(*arch["down_mp23"])
        # Level 3
        self.mp31 = EdgeMP(*arch["mp31"])
        self.mp32 = EdgeMP(*arch["mp32"])
        self.mp33 = EdgeMP(*arch["mp33"])
        self.mp34 = EdgeMP(*arch["mp34"])
        # Undown_mping 3->2
        self.up_mp32 = UpEdgeMP(*arch["up_mp32"])
        # Level 2
        self.mp221 = EdgeMP(*arch["mp221"])
        self.mp222 = EdgeMP(*arch["mp222"])
        # Undown_mping 2->1
        self.up_mp21 = UpEdgeMP(*arch["up_mp21"])
        # Level 1
        self.mp121 = EdgeMP(*arch["mp121"])
        self.mp122 = EdgeMP(*arch["mp122"])
        self.mp123 = EdgeMP(*arch["mp123"])
        self.mp124 = EdgeMP(*arch["mp124"])
        # Decoder
        self.edge_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        col  = graph.edge_index [1]
        col2 = graph.edge_index2[1]
        col3 = graph.edge_index3[1]
        # Project field along the edges (unit vector parallel to the edge)
        edge_attr  = (graph.field[col ].reshape(col .size(0), -1, 2)*graph.edgeUnitVector .unsqueeze(1)).sum(dim=-1)
        edge_attr2 = (graph.field[col2].reshape(col2.size(0), -1, 2)*graph.edgeUnitVector2.unsqueeze(1)).sum(dim=-1)
        edge_attr3 = (graph.field[col3].reshape(col3.size(0), -1, 2)*graph.edgeUnitVector3.unsqueeze(1)).sum(dim=-1)
        # Concatenate edge_attr, glob and omega
        edge_attr  = torch.cat([edge_attr , graph.glob[col ], graph.omega[col ]], dim=1)
        edge_attr2 = torch.cat([edge_attr2, graph.glob[col2], graph.omega[col2]], dim=1)
        edge_attr3 = torch.cat([edge_attr3, graph.glob[col3], graph.omega[col3]], dim=1)
        # Encode edges
        edge_attr  = F.selu(self.edge_encoder (edge_attr ))
        edge_attr2 = F.selu(self.edge_encoder2(edge_attr2))
        edge_attr3 = F.selu(self.edge_encoder3(edge_attr3))
        # Encode angles
        angle_attr   = F.selu(self.angle_encoder  (graph.angle_attr  ))
        angle_attr12 = F.selu(self.angle_encoder12(graph.angle_attr12))
        angle_attr2  = F.selu(self.angle_encoder2 (graph.angle_attr2 ))
        angle_attr23 = F.selu(self.angle_encoder23(graph.angle_attr23))
        angle_attr3  = F.selu(self.angle_encoder3 (graph.angle_attr3 ))
        # MP at level 1
        edge_attr, angle_attr = self.mp111(edge_attr, angle_attr, graph.angle_index)
        edge_attr, angle_attr = F.selu(edge_attr), F.selu(angle_attr)
        edge_attr, angle_attr = self.mp112(edge_attr, angle_attr, graph.angle_index)
        edge_attr, angle_attr = F.selu(edge_attr), F.selu(angle_attr)
        edge_attr, angle_attr = self.mp113(edge_attr, angle_attr, graph.angle_index)
        edge_attr, angle_attr = F.selu(edge_attr), F.selu(angle_attr)
        edge_attr, angle_attr = self.mp114(edge_attr, angle_attr, graph.angle_index)
        edge_attr, angle_attr = F.selu(edge_attr), F.selu(angle_attr)
        # Downsampling 1->2
        edge_attr2 = self.down_mp12(edge_attr, edge_attr2, angle_attr12, graph.angle_index12)
        edge_attr2 = F.selu(edge_attr2)
        # MP at level 2
        edge_attr2, angle_attr2 = self.mp211(edge_attr2, angle_attr2, graph.angle_index2)
        edge_attr2, angle_attr2 = F.selu(edge_attr2), F.selu(angle_attr2)
        edge_attr2, angle_attr2 = self.mp212(edge_attr2, angle_attr2, graph.angle_index2)
        edge_attr2, angle_attr2 = F.selu(edge_attr2), F.selu(angle_attr2)
        # Downsamplng 2->3
        edge_attr3 = self.down_mp23(edge_attr2, edge_attr3, angle_attr23, graph.angle_index23)
        edge_attr3 = F.selu(edge_attr3)
        # MP at level 3
        edge_attr3, angle_attr3 = self.mp31(edge_attr3, angle_attr3, graph.angle_index3)
        edge_attr3, angle_attr3 = F.selu(edge_attr3), F.selu(angle_attr3)
        edge_attr3, angle_attr3 = self.mp32(edge_attr3, angle_attr3, graph.angle_index3)
        edge_attr3, angle_attr3 = F.selu(edge_attr3), F.selu(angle_attr3)
        edge_attr3, angle_attr3 = self.mp33(edge_attr3, angle_attr3, graph.angle_index3)
        edge_attr3, angle_attr3 = F.selu(edge_attr3), F.selu(angle_attr3)
        edge_attr3,           _ = self.mp34(edge_attr3, angle_attr3, graph.angle_index3)
        edge_attr3              = F.selu(edge_attr3)
        # Upsmapling 3->2
        edge_attr2 = self.up_mp32(graph.pos, graph.y_idx_32, graph.x_idx_32, graph.weights_32,
                                  edge_attr3, graph.edge_index3, graph.edgeUnitVectorInverse3, graph.coarse_mask3,
                                  edge_attr2, graph.edge_index2, graph.edgeUnitVector2       , graph.coarse_mask2)
        edge_attr2 = F.selu(edge_attr2)
        # MP at level 2
        edge_attr2, angle_attr2 = self.mp221(edge_attr2, angle_attr2, graph.angle_index2)
        edge_attr2, angle_attr2 = F.selu(edge_attr2), F.selu(angle_attr2)
        edge_attr2,           _ = self.mp222(edge_attr2, angle_attr2, graph.angle_index2)
        edge_attr2              = F.selu(edge_attr2)
        # Upsampling 2->1
        edge_attr = self.up_mp21(graph.pos, graph.y_idx_21, graph.x_idx_21, graph.weights_21,
                                 edge_attr2, graph.edge_index2, graph.edgeUnitVectorInverse2, graph.coarse_mask2,
                                 edge_attr , graph.edge_index , graph.edgeUnitVector)
        edge_attr = F.selu(edge_attr)
        # MP at level 1
        edge_attr, angle_attr = self.mp121(edge_attr, angle_attr, graph.angle_index)
        edge_attr, angle_attr = F.selu(edge_attr), F.selu(angle_attr)
        edge_attr, angle_attr = self.mp122(edge_attr, angle_attr, graph.angle_index)
        edge_attr, angle_attr = F.selu(edge_attr), F.selu(angle_attr)
        edge_attr, angle_attr = self.mp123(edge_attr, angle_attr, graph.angle_index)
        edge_attr, angle_attr = F.selu(edge_attr), F.selu(angle_attr)
        edge_attr,          _ = self.mp124(edge_attr, angle_attr, graph.angle_index)
        edge_attr             = F.selu(edge_attr)
        # Decode
        edge_attr = self.edge_decoder(edge_attr) # |E|x1
        # Edge scalars to node vectors
        output = edgeScalarToNodeVector(edge_attr, graph.edge_index, edgeUnitVectorInverse=graph.edgeUnitVectorInverse) # [|V|x2]
        # Time-step
        return graph.field[:,-self.num_fields:] + output