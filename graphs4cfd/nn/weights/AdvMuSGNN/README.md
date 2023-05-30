Parameters of the MuS-GNN models for simulating advection from Lino et al. (2022) ([https://doi.org/10.1063/5.0097679](https://doi.org/10.1063/5.0097679)).

The models included are:
- 1S-GNN: `AdvOneScaleGNN.chk`. Load with: `graphs4cfd.nn.AdvOneScaleGNN(checkpoint='AdvOneScaleGNN.chk')`
- 2S-GNN: `AdvTwoScaleGNN.chk`. Load with: `graphs4cfd.nn.AdvTwoScaleGNN(checkpoint='AdvTwoScaleGNN.chk')`
- 3S-GNN: `AdvThreeScaleGNN.chk`. Load with: `graphs4cfd.nn.AdvThreeScaleGNN(checkpoint='AdvThreeScaleGNN.chk')`
- 4S-GNN: `AdvFourScaleGNN.chk`. Load with: `graphs4cfd.nn.AdvFourScaleGNN(checkpoint='AdvFourScaleGNN.chk')`