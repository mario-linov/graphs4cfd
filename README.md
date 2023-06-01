# Graphs4CFD

![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)

<p align="center">
  <img src="https://i.ibb.co/BnV3P44/example-remus-gnn.gif" />
</p>

**Graphs4CFD** is a library built upon [PyTorch](https://pytorch.org/) and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) (PyG) to code and train Graph Neural Networks (GNNs) based solvers for Computational Fluid Dynamics (CFD) applications.

## Contents
<!-- Table of contents -->
- [Implemented GNN models](#implemented-gnn-models)
- [Installation](#installation)
- [Examples](#examples)
- [Cite](#cite)

## Implemented GNN models

To date, Graphs4CFD supports the following GNN models:
- MuS-GNN - Lino et al. 2022 ([https://doi.org/10.1063/5.0097679](https://doi.org/10.1063/5.0097679))
- REMuS-GNN - Lino et al. 2022 ([https://doi.org/10.1063/5.0097679](https://doi.org/10.1063/5.0097679))
- Mult-scale GNNs with low-resolution graphs obtained by Guillard's coarsening - Appendix C.3 in Lino et al. 2022 ([https://doi.org/10.1063/5.0097679](https://doi.org/10.1063/5.0097679))

## Installation

Graphs4CFD requires Python 3.7 or higher and a version of [PyTorch](https://pytorch.org/) compatible with your CUDA version.
We recomend installing Graphs4CFD and its dependencies in a virtual envioroment, e.g., using [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
To install Graphs4CFD and its dependecies (except PyTorch), run the following commands:

```bash
git clone git@github.com:mario-linov/graphs4cfd.git
cd graphs4cfd
pip intall -e .
```

This also installs [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and compiles and installs [PyTorch Cluster](https://github.com/rusty1s/pytorch_cluster), so it may take a while.
Once Graphs4CFD has been installed, it can be imported in Python as follows:

```python
import graphs4cfd as gfd
```

## Examples

There are examples of how to use Graphs4CFD for training and inference in the [examples/](https://github.com/mario-linov/graphs4cfd/tree/main/examples) folder.

:warning:To clone the jupyter notebooks in the [inference/](https://github.com/mario-linov/graphs4cfd/tree/main/examples/inference) folder you need to have [Git LFS](https://git-lfs.com/) installed.
On Ubuntu, Git LFS can be installed painlessly by running:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

## Cite

To cite Graphs4CFD, please use the following reference:

Mario Lino, Stathi Fotiadis, Anil A. Bharath, and Chris Cantwell. "Multi-scale rotation-equivariant graph neural networks for unsteady Eulerian fluid dynamics". Physics of Fluids, 34 (2022).

```latex
@article{lino2022multi,
    author = {Lino, Mario and Fotiadis, Stathi and Bharath, Anil A. and Cantwell, Chris},
    title = {{Multi-scale rotation-equivariant graph neural networks for unsteady Eulerian fluid dynamics}},
    journal = {Physics of Fluids},
    volume = {34},
    year = {2022},
    url = {https://doi.org/10.1063/5.0097679},
}
```
