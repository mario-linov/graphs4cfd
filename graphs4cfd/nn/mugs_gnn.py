import os
import torch
import torch.nn.functional as F
from typing import Optional

from .model import GNN
from .blocks import MLP, MP, restriction, knn_interpolate
from ..graph import Graph


class NsTwoGuillardScaleGNN(GNN):
    """ Two-scale GNN with the low-resolution graph otained by Guillard's coarsening.
    See Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder" : (2, (128,128,128), False),
        "edge_encoder2": (2, (128,128,128), False),
        "node_encoder" : (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 2
        "mp21": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp22": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp23": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp24": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 1
        "mp121": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    } 
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "2GS-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuGSGNN/NsTwoGuillardScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder  = MLP(*arch["edge_encoder" ])
        self.edge_encoder2 = MLP(*arch["edge_encoder2"])
        self.node_encoder  = MLP(*arch["node_encoder" ])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Level 2
        self.mp21 = MP(*arch["mp21"])
        self.mp22 = MP(*arch["mp22"])
        self.mp23 = MP(*arch["mp23"])
        self.mp24 = MP(*arch["mp24"])
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None):
        num_nodes = graph.num_nodes
        field, edge_attr, edge_attr_2 = graph.field, graph.edge_attr, graph.edge_attr2
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr  = F.selu(self.edge_encoder (graph.edge_attr ))
        graph.edge_attr2 = F.selu(self.edge_encoder2(graph.edge_attr2))
        graph.field      = F.selu(self.node_encoder (graph.field     ))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1, batch1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr, graph.batch
        # Downsampling to level 2
        graph.field = graph.field[graph.coarse_mask2]
        graph.pos   = graph.pos  [graph.coarse_mask2]
        graph.batch = graph.batch[graph.coarse_mask2]
        restriction(graph, graph.coarse_mask2, graph.edge_attr2, graph.edge_index2, num_nodes, self.device)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp21(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp22(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp23(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp24(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 1
        graph.field = knn_interpolate(graph.field, graph.y_idx_21, graph.x_idx_21, graph.weights_21)
        graph.field = torch.cat([graph.field, field1], dim=1)
        graph.pos, graph.edge_index, graph.edge_attr, graph.batch = pos1, edge_index1, edge_attr1, batch1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp123(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp124(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr, graph.edge_attr2 = field, edge_attr, edge_attr_2
        # Time-step
        return graph.field[:,-self.num_fields:] + output
    

class NsThreeGuillardScaleGNN(GNN):
    """ Three-scale GNN with the low-resolution graphs otained by Guillard's coarsening.
    See Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder" : (2, (128,128,128), False),
        "edge_encoder2": (2, (128,128,128), False),
        "edge_encoder3": (2, (128,128,128), False),
        "node_encoder" : (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 3
        "mp31": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp32": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp33": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp34": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 2
        "mp221": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 1
        "mp121": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "3GS-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuGSGNN/NsThreeGuillardScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch):
        self.arch = arch
        # Encoder
        self.edge_encoder  = MLP(*arch["edge_encoder" ])
        self.edge_encoder2 = MLP(*arch["edge_encoder2"])
        self.edge_encoder3 = MLP(*arch["edge_encoder3"])
        self.node_encoder  = MLP(*arch["node_encoder" ])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Level 3
        self.mp31  = MP(*arch["mp31"])
        self.mp32  = MP(*arch["mp32"])
        self.mp33  = MP(*arch["mp33"])
        self.mp34  = MP(*arch["mp34"])
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, data, t):
        num_nodes = data.num_nodes
        field, edge_attr, edge_attr_2, edge_attr_3 = data.field, data.edge_attr, data.edge_attr2, data.edge_attr3
        # Concatenate field, loc, glob and omega
        data.field = torch.cat([getattr(data, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(data, v)], dim=1)
        # Encode
        data.edge_attr  = F.selu(self.edge_encoder (data.edge_attr ))
        data.edge_attr2 = F.selu(self.edge_encoder2(data.edge_attr2))
        data.edge_attr3 = F.selu(self.edge_encoder3(data.edge_attr3))
        data.field      = F.selu(self.node_encoder (data.field     ))
        # MP at level 1
        data.field, data.edge_attr = self.mp111(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp112(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp113(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp114(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        field1, pos1, edge_index1, edge_attr1, batch1 = data.field, data.pos, data.edge_index, data.edge_attr, data.batch
        # Downsampling to level 2
        data.field = data.field[data.coarse_mask2]
        data.pos   = data.pos  [data.coarse_mask2]
        data.batch = data.batch[data.coarse_mask2]
        restriction(data, data.coarse_mask2, data.edge_attr2, data.edge_index2, num_nodes, self.device)
        # MP at level 2
        data.field, data.edge_attr = self.mp211(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp212(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        field2, pos2, edge_index2, edge_attr2, batch2 = data.field, data.pos, data.edge_index, data.edge_attr, data.batch
        # Downsampling to level 3
        mask = data.coarse_mask3[data.coarse_mask2]
        data.field = data.field[mask]
        data.pos   = data.pos  [mask]
        data.batch = data.batch[mask]
        restriction(data, data.coarse_mask3, data.edge_attr3, data.edge_index3, num_nodes, self.device)
        # MP at level 3
        data.field, data.edge_attr = self.mp31(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp32(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp33(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, _              = self.mp34(data.field, data.edge_attr, data.edge_index)
        data.field                 = F.selu(data.field)
        # Upsampling to level 2
        data.field = knn_interpolate(data.field, data.y_idx_32, data.x_idx_32, data.weights_32)
        # data.field = knn_interpolate(data.field, data.pos, pos2, batch_x=data.batch, batch_y=batch2, k=6)
        data.field = torch.cat([data.field, field2], dim=1)
        data.pos, data.edge_index, data.edge_attr, data.batch = pos2, edge_index2, edge_attr2, batch2
        # MP at level 2
        data.field, data.edge_attr = self.mp221(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, _              = self.mp222(data.field, data.edge_attr, data.edge_index)
        data.field                 = F.selu(data.field)
        # Upsampling to level 1
        data.field = knn_interpolate(data.field, data.y_idx_21, data.x_idx_21, data.weights_21)
        # data.field = knn_interpolate(data.field, data.pos, pos1, batch_x=data.batch, batch_y=batch1, k=6)
        data.field = torch.cat([data.field, field1], dim=1)
        data.pos, data.edge_index, data.edge_attr, data.batch = pos1, edge_index1, edge_attr1, batch1
        # MP at level 1
        data.field, data.edge_attr = self.mp121(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp122(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp123(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, _              = self.mp124(data.field, data.edge_attr, data.edge_index)
        data.field                 = F.selu(data.field)
        # Decode
        output = self.node_decoder(data.field)
        # Restore data
        data.field, data.edge_attr, data.edge_attr2, data.edge_attr3 = field, edge_attr, edge_attr_2, edge_attr_3
        # Time-step
        return data.field[:,-self.num_fields:] + output
    

class NsFourGuillardScaleGNN(GNN):
    """ Four-scale GNN with the low-resolution graphs otained by Guillard's coarsening.
    See Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    ```python
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder" : (2, (128,128,128), False),
        "edge_encoder2": (2, (128,128,128), False),
        "edge_encoder3": (2, (128,128,128), False),
        "edge_encoder4": (2, (128,128,128), False),
        "node_encoder" : (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 3
        "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 4
        "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 3
        "mp321": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
        "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 2
        "mp221": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Level 1
        "mp121": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    ```

    Args:
        model (str, optional): Name of the model to load. Defaults to None.
    """

    def __init__(self, model: str = None, *args, **kwargs):
        if model is not None:
            if model == "4GS-GNN-NsCircle-v1":
                super().__init__(arch=None, weights=None, checkpoint=os.path.join(os.path.dirname(__file__), 'weights/NsMuGSGNN/NsFourGuillardScaleGNN.chk'), *args, **kwargs)
            else:
                raise ValueError(f"Model {model} not recognized.")
        else:
            super().__init__(*args, **kwargs)

    def load_arch(self, arch):
        self.arch = arch
        # Encoder
        self.edge_encoder  = MLP(*arch["edge_encoder" ])
        self.edge_encoder2 = MLP(*arch["edge_encoder2"])
        self.edge_encoder3 = MLP(*arch["edge_encoder3"])
        self.edge_encoder4 = MLP(*arch["edge_encoder4"])
        self.node_encoder  = MLP(*arch["node_encoder" ])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Level 3
        self.mp311  = MP(*arch["mp311"])
        self.mp312  = MP(*arch["mp312"])
        # Level 4
        self.mp41  = MP(*arch["mp41"])
        self.mp42  = MP(*arch["mp42"])      
        self.mp43  = MP(*arch["mp43"])
        self.mp44  = MP(*arch["mp44"])
        # Level 3
        self.mp321  = MP(*arch["mp321"])
        self.mp322  = MP(*arch["mp322"])
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, data, t):
        num_nodes = data.num_nodes
        field, edge_attr, edge_attr_2, edge_attr_3, edge_attr_4 = data.field, data.edge_attr, data.edge_attr2, data.edge_attr3, data.edge_attr4
        # Concatenate field, loc, glob and omega
        data.field = torch.cat([getattr(data, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(data, v)], dim=1)
        # Encode
        data.edge_attr  = F.selu(self.edge_encoder (data.edge_attr ))
        data.edge_attr2 = F.selu(self.edge_encoder2(data.edge_attr2))
        data.edge_attr3 = F.selu(self.edge_encoder3(data.edge_attr3))
        data.edge_attr4 = F.selu(self.edge_encoder4(data.edge_attr4))
        data.field      = F.selu(self.node_encoder (data.field     ))
        # MP at level 1
        data.field, data.edge_attr = self.mp111(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp112(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp113(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp114(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        field1, pos1, edge_index1, edge_attr1, batch1 = data.field, data.pos, data.edge_index, data.edge_attr, data.batch
        # Downsampling to level 2
        data.field = data.field[data.coarse_mask2]
        data.pos   = data.pos  [data.coarse_mask2]
        data.batch = data.batch[data.coarse_mask2]
        restriction(data, data.coarse_mask2, data.edge_attr2, data.edge_index2, num_nodes, self.device)
        # MP at level 2
        data.field, data.edge_attr = self.mp211(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp212(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        field2, pos2, edge_index2, edge_attr2, batch2 = data.field, data.pos, data.edge_index, data.edge_attr, data.batch
        # Downsampling to level 3
        mask = data.coarse_mask3[data.coarse_mask2]
        data.field = data.field[mask]
        data.pos   = data.pos  [mask]
        data.batch = data.batch[mask]
        restriction(data, data.coarse_mask3, data.edge_attr3, data.edge_index3, num_nodes, self.device)
        # MP at level 3
        data.field, data.edge_attr = self.mp311(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp312(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        field3, pos3, edge_index3, edge_attr3, batch3 = data.field, data.pos, data.edge_index, data.edge_attr, data.batch
        # Downsampling to level 4
        mask = data.coarse_mask4[data.coarse_mask3]
        data.field = data.field[mask]
        data.pos   = data.pos  [mask]
        data.batch = data.batch[mask]
        restriction(data, data.coarse_mask4, data.edge_attr4, data.edge_index4, num_nodes, self.device)
        # MP at level 4
        data.field, data.edge_attr = self.mp41(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp42(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp43(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, _              = self.mp44(data.field, data.edge_attr, data.edge_index)
        data.field                 = F.selu(data.field)
        # Upsampling to level 3
        data.field = knn_interpolate(data.field, data.y_idx_43, data.x_idx_43, data.weights_43)
        data.field = torch.cat([data.field, field3], dim=1)
        data.pos, data.edge_index, data.edge_attr, data.batch = pos3, edge_index3, edge_attr3, batch3
        # MP at level 3
        data.field, data.edge_attr = self.mp321(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, _              = self.mp322(data.field, data.edge_attr, data.edge_index)
        data.field                 = F.selu(data.field)
        # Upsampling to level 2
        data.field = knn_interpolate(data.field, data.y_idx_32, data.x_idx_32, data.weights_32)
        data.field = torch.cat([data.field, field2], dim=1)
        data.pos, data.edge_index, data.edge_attr, data.batch = pos2, edge_index2, edge_attr2, batch2
        # MP at level 2
        data.field, data.edge_attr = self.mp221(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, _              = self.mp222(data.field, data.edge_attr, data.edge_index)
        data.field                 = F.selu(data.field)
        # Upsampling to level 1
        data.field = knn_interpolate(data.field, data.y_idx_21, data.x_idx_21, data.weights_21)
        data.field = torch.cat([data.field, field1], dim=1)
        data.pos, data.edge_index, data.edge_attr, data.batch = pos1, edge_index1, edge_attr1, batch1
        # MP at level 1
        data.field, data.edge_attr = self.mp121(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp122(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, data.edge_attr = self.mp123(data.field, data.edge_attr, data.edge_index)
        data.field, data.edge_attr = F.selu(data.field), F.selu(data.edge_attr)
        data.field, _              = self.mp124(data.field, data.edge_attr, data.edge_index)
        data.field                 = F.selu(data.field)
        # Decode
        output = self.node_decoder(data.field)
        # Restore data
        data.field, data.edge_attr, data.edge_attr2, data.edge_attr3, data.edge_attr4 = field, edge_attr, edge_attr_2, edge_attr_3, edge_attr_4
        # Time-step
        return data.field[:,-self.num_fields:] + output