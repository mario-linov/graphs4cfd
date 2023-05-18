import torch
import torch.nn.functional as F
from typing import Optional

from .model import Model
from .blocks import MLP, MP, DownMP, UpMP
from ..graph import Graph


class NsOneScaleGNN(Model):
    """The 1S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp11": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp12": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp13": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp14": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp15": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp16": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp17": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp18": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp11 = MP(*arch["mp11"])
        self.mp12 = MP(*arch["mp12"])
        self.mp13 = MP(*arch["mp13"])
        self.mp14 = MP(*arch["mp14"])
        self.mp15 = MP(*arch["mp15"])
        self.mp16 = MP(*arch["mp16"])
        self.mp17 = MP(*arch["mp17"])
        self.mp18 = MP(*arch["mp18"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp11(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp12(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp13(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp14(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp15(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp16(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp17(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp18(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class NsTwoScaleGNN(Model):
    """The 2S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": ((2+128, (128,128,128), True), 0.15),
        # Level 2
        "mp21": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp22": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp23": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp24": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": ((2+128+128, (128,128,128), True), 0.15),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12 = DownMP(*arch["down_mp12"], 1)
        # Level 2
        self.mp21 = MP(*arch["mp21"])
        self.mp22 = MP(*arch["mp22"])
        self.mp23 = MP(*arch["mp23"])
        self.mp24 = MP(*arch["mp24"])
        # Upsampling to level 1
        self.up_mp21 = UpMP(*arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
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
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
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
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class NsThreeScaleGNN(Model):
    """The 3S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": ((2+128, (128,128,128), True), 0.15),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": ((2+128, (128,128,128), True), 0.30),
        # Level 3
        "mp31": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp32": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp33": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp34": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": ((2+128+128, (128,128,128), True), 0.30),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": ((2+128+128, (128,128,128), True), 0.15),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }   
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"][0], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Downsampling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"][0], 2)
        # Level 3
        self.mp31  = MP(*arch["mp31"])
        self.mp32  = MP(*arch["mp32"])
        self.mp33  = MP(*arch["mp33"])
        self.mp34  = MP(*arch["mp34"])
        # Upsampling to level 2
        self.up_mp32 = UpMP(arch["up_mp32"][0], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Upsampling to level 1
        self.up_mp21 = UpMP(arch["up_mp21"][0], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:  
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp31(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp32(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp33(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp34(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
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
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output
    

class NsFourScaleGNN(Model):
    """The 4S-GNN for incompressible flow inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (5, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": ((2+128, (128,128,128), True), 0.15),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": ((2+128, (128,128,128), True), 0.30),
        # Level 3
        "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp34": ((2+128, (128,128,128), True), 0.60),
        # Level 4
        "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp43": ((2+128+128, (128,128,128), True), 0.60),
        # Level 3
        "mp321": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": ((2+128+128, (128,128,128), True), 0.30),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": ((2+128+128, (128,128,128), True), 0.15),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,3), False),
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp113 = MP(*arch["mp113"])
        self.mp114 = MP(*arch["mp114"])
        # Downsampling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"][0], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Downsampling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"][0], 2)
        # Level 3
        self.mp311 = MP(*arch["mp311"])
        self.mp312 = MP(*arch["mp312"])
        # Downsampling to level 4
        self.down_mp34 = DownMP(arch["down_mp34"][0], 3)
        # Level 4
        self.mp41 = MP(*arch["mp41"])
        self.mp42 = MP(*arch["mp42"])
        self.mp43 = MP(*arch["mp43"])
        self.mp44 = MP(*arch["mp44"])
        # Upsampling to level 3
        self.up_mp43 = UpMP(arch["up_mp43"][0], 4)
        # Level 3
        self.mp321  = MP(*arch["mp321"])
        self.mp322  = MP(*arch["mp322"])
        # Upsampling to level 2
        self.up_mp32 = UpMP(arch["up_mp32"][0], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Upsampling to level 1
        self.up_mp21 = UpMP(arch["up_mp21"][0], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        self.mp123 = MP(*arch["mp123"])
        self.mp124 = MP(*arch["mp124"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp113(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp114(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp311(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp312(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field3, pos3, edge_index3, edge_attr3 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Downsampling to level 4
        graph = self.down_mp34(graph, activation=torch.tanh)
        # Level 4
        graph.field, graph.edge_attr = self.mp41(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp42(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp43(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp44(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 3
        graph = self.up_mp43(graph, field3, pos3, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index3, edge_attr3
        # Level 3
        graph.field, graph.edge_attr = self.mp321(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp322(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Upsampling to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
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
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output
    


class AdvOneScaleGNN(Model):
    """The 1S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class AdvTwoScaleGNN(Model):
    """The 2S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": ((2+128, (128,128,128), True), 0.02),
        # Level 2
        "mp21": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp22": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp23": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp24": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": ((2+128+128, (128,128,128), True), 0.02),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        # Pooling to level 2
        self.down_mp12 = DownMP(*arch["down_mp12"], 1)
        # Level 2
        self.mp21 = MP(*arch["mp21"])
        self.mp22 = MP(*arch["mp22"])
        self.mp23 = MP(*arch["mp23"])
        self.mp24 = MP(*arch["mp24"])
        # Undown_mping to level 1
        self.up_mp21 = UpMP(*arch["up_mp21"], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp21(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp22(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp23(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp24(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class AdvThreeScaleGNN(Model):
    """The 3S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": ((2+128, (128,128,128), True), 0.02),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": ((2+128, (128,128,128), True), 0.04),
        # Level 3
        "mp31": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp32": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp33": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp34": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": ((2+128+128, (128,128,128), True), 0.04),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": ((2+128+128, (128,128,128), True), 0.02),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        # Pooling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"][0], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Pooling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"][0], 2)
        # Level 3
        self.mp31  = MP(*arch["mp31"])
        self.mp32  = MP(*arch["mp32"])
        self.mp33  = MP(*arch["mp33"])
        self.mp34  = MP(*arch["mp34"])
        # Undown_mping to level 2
        self.up_mp32 = UpMP(arch["up_mp32"][0], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Undown_mping to level 1
        self.up_mp21 = UpMP(arch["up_mp21"][0], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp31(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp32(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp33(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp34(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output


class AdvFourScaleGNN(Model):
    """The 4S-GNN for advection inference from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

    In that work, the hyperparameters were:
    arch = {
        ################ Edge-functions ################## Node-functions ##############
        # Encoder
        "edge_encoder": (2, (128,128,128), False),
        "node_encoder": (4, (128,128,128), False),
        # Level 1
        "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp12": ((2+128, (128,128,128), True), 0.02),
        # Level 2
        "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp23": ((2+128, (128,128,128), True), 0.04),
        # Level 3
        "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "down_mp34": ((2+128, (128,128,128), True), 0.08),
        # Level 4
        "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp43": ((2+128+128, (128,128,128), True), 0.08),
        # Level 3
        "mp321": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp32": ((2+128+128, (128,128,128), True), 0.04),
        # Level 2
        "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "up_mp21": ((2+128+128, (128,128,128), True), 0.02),
        # Level 1
        "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
        # Decoder
        "decoder": (128, (128,128,1), False),
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Encoder
        self.edge_encoder = MLP(*arch["edge_encoder"])
        self.node_encoder = MLP(*arch["node_encoder"])
        # Level 1
        self.mp111 = MP(*arch["mp111"])
        self.mp112 = MP(*arch["mp112"])
        # Pooling to level 2
        self.down_mp12 = DownMP(arch["down_mp12"][0], 1)
        # Level 2
        self.mp211 = MP(*arch["mp211"])
        self.mp212 = MP(*arch["mp212"])
        # Pooling to level 3
        self.down_mp23 = DownMP(arch["down_mp23"][0], 2)
        # Level 3
        self.mp311 = MP(*arch["mp311"])
        self.mp312 = MP(*arch["mp312"])
        # Pooling to level 4
        self.down_mp34 = DownMP(arch["down_mp34"][0], 3)
        # Level 4
        self.mp41 = MP(*arch["mp41"])
        self.mp42 = MP(*arch["mp42"])
        self.mp43 = MP(*arch["mp43"])
        self.mp44 = MP(*arch["mp44"])
        # Undown_mping to level 3
        self.up_mp43 = UpMP(arch["up_mp43"][0], 4)
        # Level 3
        self.mp321  = MP(*arch["mp321"])
        self.mp322  = MP(*arch["mp322"])
        # Undown_mping to level 2
        self.up_mp32 = UpMP(arch["up_mp32"][0], 3)
        # Level 2
        self.mp221 = MP(*arch["mp221"])
        self.mp222 = MP(*arch["mp222"])
        # Undown_mping to level 1
        self.up_mp21 = UpMP(arch["up_mp21"][0], 2)
        # Level 1
        self.mp121 = MP(*arch["mp121"])
        self.mp122 = MP(*arch["mp122"])
        # Decoder
        self.node_decoder = MLP(*arch["decoder"])
        self.to(self.device)

    def forward(self, graph: Graph, t: Optional[int] = None) -> torch.Tensor:
        field, edge_attr = graph.field, graph.edge_attr
        # Concatenate field, loc, glob and omega
        graph.field = torch.cat([getattr(graph, v) for v in ('field', 'loc', 'glob', 'omega') if hasattr(graph, v)], dim=1)
        # Encode
        graph.edge_attr = F.selu(self.edge_encoder(graph.edge_attr))
        graph.field     = F.selu(self.node_encoder(graph.field))
        # MP at level 1
        graph.field, graph.edge_attr = self.mp111(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp112(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field1, pos1, edge_index1, edge_attr1 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 2
        graph = self.down_mp12(graph, activation=torch.tanh)
        # MP at level 2
        graph.field, graph.edge_attr = self.mp211(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp212(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field2, pos2, edge_index2, edge_attr2 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 3
        graph = self.down_mp23(graph, activation=torch.tanh)
        # MP at level 3
        graph.field, graph.edge_attr = self.mp311(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp312(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        field3, pos3, edge_index3, edge_attr3 = graph.field, graph.pos, graph.edge_index, graph.edge_attr
        # Pooling to level 4
        graph = self.down_mp34(graph, activation=torch.tanh)
        # Level 4
        graph.field, graph.edge_attr = self.mp41(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp42(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, graph.edge_attr = self.mp43(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp44(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 3
        graph = self.up_mp43(graph, field3, pos3, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index3, edge_attr3
        # Level 3
        graph.field, graph.edge_attr = self.mp321(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp322(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 2
        graph = self.up_mp32(graph, field2, pos2, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index2, edge_attr2
        # MP at level 2
        graph.field, graph.edge_attr = self.mp221(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp222(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Undown_mping to level 1
        graph = self.up_mp21(graph, field1, pos1, activation=torch.tanh)
        graph.edge_index, graph.edge_attr = edge_index1, edge_attr1
        # MP at level 1
        graph.field, graph.edge_attr = self.mp121(graph.field, graph.edge_attr, graph.edge_index)
        graph.field, graph.edge_attr = F.selu(graph.field), F.selu(graph.edge_attr)
        graph.field, _              = self.mp122(graph.field, graph.edge_attr, graph.edge_index)
        graph.field                 = F.selu(graph.field)
        # Decode
        output = self.node_decoder(graph.field)
        # Restore data
        graph.field, graph.edge_attr = field, edge_attr
        # Time-step
        return graph.field[:,-self.num_fields:] + output