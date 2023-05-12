import torch

from ..graph import Graph


class AddUniformNoise():
    r"""Transformation class to add uniform noise, with amplitude \epsilon, to the input fields of a graph.
    Given a graph with input fields $u_i$, the transformation is given by
    $$u_i \leftarrow u_i + U_{[-\epsilon, \epsilon]}$$
    for each field $u_i$ and where $U_{[-\epsilon, \epsilon]}$ is a uniform random variable in the interval $[-\epsilon, \epsilon]$.

    Args:
        eps (float): Amplitude of the noise.

    Methods:
        __call__(Graph) -> Graph: Adds uniform noise, with amplitude \epsilon, to the input fields of a graph.
    
    """

    def __init__(self, eps: float):
        self.eps = eps

    def __call__(self, graph: Graph) -> Graph:
        graph.field += self.eps*( 2*torch.rand_like(graph.field) - 1 )
        return graph