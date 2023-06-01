# Examples

This folder contains examples of how to use Graphs4CFD for **training** and **inference**.

## Training examples

The following training scripts are available:
- [AdvMuSGNN/](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/AdvMuSGNN)
    - [AdvOneScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/AdvMuSGNN/AdvOneScaleGNN.py) Trains a MuS-GNN model with a single scale against uniform advection on rectangular domains.
    - [AdvTwoScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/AdvMuSGNN/AdvTwoScaleGNN.py) Trains a MuS-GNN model with two scales against uniform advection on rectangular domains.
    - [AdvThreeScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/AdvMuSGNN/AdvThreeScaleGNN.py) Trains a MuS-GNN model with three scales against uniform advection on rectangular domains.
    - [AdvFourScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/AdvMuSGNN/AdvFourScaleGNN.py) Trains a MuS-GNN model with four scales against uniform advection on rectangular domains.
- [NsMusGNN/](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuSGNN)
    - [NsOneScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuSGNN/NsOneScaleGNN.py) Trains a MuS-GNN model with a single scale against the flow around a circular cylinder as described by the incompressible Navier-Stokes equations.
    - [NsTwoScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuSGNN/NsTwoScaleGNN.py) Trains a MuS-GNN model with two scales against the flow around a circular cylinder as described by the incompressible Navier-Stokes equations.
    - [NsThreeScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuSGNN/NsThreeScaleGNN.py) Trains a MuS-GNN model with three scales against the flow around a circular cylinder as described by the incompressible Navier-Stokes equations.
    - [NsFourScaleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuSGNN/NsFourScaleGNN.py) Trains a MuS-GNN model with four scales against the flow around a circular cylinder as described by the incompressible Navier-Stokes equations.
- [NsMuGSGNN/](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuGSGNN)
    - [NsTwoGuillardSacleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuGSGNN/NsTwoGuillardSacleGNN.py) Trains a MuGS-GNN model with two scales against the flow around a circular cylinder as described by the incompressible Navier-Stokes equations.
    - [NsThreeGuillardSacleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuGSGNN/NsThreeGuillardSacleGNN.py) Trains a MuGS-GNN model with three scales against the flow around a circular cylinder as described by the incompressible Navier-Stokes equations.
    - [NsFourGuillardSacleGNN.py](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsMuGSGNN/NsFourGuillardSacleGNN.py) Trains a MuGS-GNN model with four scales against the flow around a circular cylinder as described by the incompressible Navier-Stokes equations.
- [NsREMuSGNN/](https://github.com/mario-linov/graphs4cfd/tree/main/examples/training/NsREMuSGNN)
    - [NsRotEquiTreeScaleGNN.py](https://github.com/mario-linov/.pygraphs4cfd/tree/main/examples/training/NsREMuSGNN/NsRotEquiTreeScaleGNN.py) Trains a REMuS-GNN (rotation-equivariant version of the MuS-GNN) model with three scales against the flow around an elliptical cylinder as described by the incompressible Navier-Stokes equations.


## Inference examples

:warning: To clone the jupyter notebooks in the [inference/](https://github.com/mario-linov/graphs4cfd/tree/main/examples/inference) folder you need to have [Git LFS](https://git-lfs.com/) installed.
On Ubuntu, Git LFS can be installed painlessly by running:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

The following inference notebooks are available:
- [mus_gnn/adv_mus_gnn.ipynb](https://github.com/mario-linov/graphs4cfd/tree/main/examples/inference/mus_gnn/adv_mus_gnn.ipynb) Demonstrates how to use trained MuS-GNN models to predict the solution of the advection equation on different domains.
- [mus_gnn/ns_mus_gnn.ipynb](https://github.com/mario-linov/graphs4cfd/tree/main/examples/inference/mus_gnn/ns_mus_gnn.ipynb) Demonstrates how to use trained MuS-GNN models to predict the solution of the incompressible Navier-Stokes for the flow around a circular cylinder.
- [mugs_gnn/ns_mugs_gnn.ipynb](https://github.com/mario-linov/graphs4cfd/tree/main/examples/inference/mugs_gnn/ns_mugs_gnn.ipynb) Demonstrates how to use trained MuGS-GNN models to predict the solution of the incompressible Navier-Stokes for the flow around a circular cylinder.
- [remus_gnn/ns_remus_gnn.ipynb](https://github.com/mario-linov/graphs4cfd/tree/main/examples/inference/remus_gnn/ns_remus_gnn.ipynb) Demonstrates how to use trained REMuS-GNN models to predict the solution of the incompressible Navier-Stokes for the flow around an elliptical cylinder.