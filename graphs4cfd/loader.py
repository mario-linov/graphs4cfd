from typing import Union, List, Optional
import torch.utils.data
from torch_geometric.data import Data, Dataset, Batch
from torchvision import transforms


class Collater(object):
    """ A modified version of PyTorch Geometric's default collate function to handle REMuS-GNNs and add transformations applied to the whole
    batch instead of individual graphs."""

    def __init__(self, transform=None):
        self.transform = transform

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data) or isinstance(elem):
            # Apply correction for angle index in edge-angle formulation 
            if hasattr(elem, 'angle_index'):
                num_nodes = elem.num_nodes
                num_edges = elem.edge_index.size(1)
                for graph in batch[1:]:
                    graph.angle_index += (num_edges-num_nodes)
                    if hasattr(elem, 'angle_index12'): graph.angle_index12[0] += (num_edges-num_nodes)
                    num_nodes += graph.num_nodes
                    num_edges += graph.num_edges
            if hasattr(elem, 'angle_index2'):
                num_nodes = elem.num_nodes
                num_edges = elem.edge_index2.size(1)
                for graph in batch[1:]:
                    graph.angle_index2 += (num_edges-num_nodes)
                    graph.angle_index12[1] += (num_edges-num_nodes)
                    if hasattr(elem, 'angle_index23'): graph.angle_index23[0] += (num_edges-num_nodes)
                    num_nodes += graph.num_nodes
                    num_edges += graph.edge_index2.size(1)
            if hasattr(elem, 'angle_index3'):
                num_nodes = elem.num_nodes
                num_edges = elem.edge_index3.size(1)
                for graph in batch[1:]:
                    graph.angle_index3 += (num_edges-num_nodes)
                    graph.angle_index23[1] += (num_edges-num_nodes)
                    if hasattr(elem, 'angle_index34'): graph.angle_index34[0] += (num_edges-num_nodes)
                    num_nodes += graph.num_nodes
                    num_edges += graph.edge_index3.size(1)
            if hasattr(elem, 'angle_index4'):
                num_nodes = elem.num_nodes
                num_edges = elem.edge_index4.size(1)
                for graph in batch[1:]:
                    graph.angle_index4 += (num_edges-num_nodes)
                    graph.angle_index34[1] += (num_edges-num_nodes)
                    num_nodes += graph.num_nodes
                    num_edges += graph.edge_index4.size(1)
            # Create batch
            batch = Batch.from_data_list(batch)
            # Apply transforms
            if self.transform is None:
                return batch
            else:
                return self.transform(batch)

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[Data]],
        batch_size: int = 1,
        shuffle: bool = False,
        transform: Optional[transforms.Compose] = None,
        **kwargs,
    ):
        if "collate_fn" in kwargs: del kwargs["collate_fn"]
        # Add transforms to collate_fn
        collate_fn = Collater(transform)       
        super().__init__(dataset, batch_size, shuffle, collate_fn=collate_fn, **kwargs)