from torch_geometric.data import Data

from .plot import pos, field, pos_field


class Graph(Data):
    r"""A data object describing a graph. Same as torch_geometric.data.Data but with some plotting methods."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_pos(self, **kwargs):
        pos(self.pos, **kwargs)

    def plot_field(self, *args, **kwargs):
        field(self.pos, bound=getattr(self, 'bound') if hasattr(self, 'bound') else None, *args, **kwargs)

    def plot_pos_field(self, *args, **kwargs):
        pos_field(self.pos, *args, **kwargs)