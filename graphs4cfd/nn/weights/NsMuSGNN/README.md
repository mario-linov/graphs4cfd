Parameters of the MuS-GNN models for simulating incompressible flow pass a vertical array of vertical cylinders from Lino et al. (2022) ([https://doi.org/10.1063/5.0097679](https://doi.org/10.1063/5.0097679)).

The models included are:
- 1S-GNN: `NsOneScaleGNN.chk`. Load with: `graphs4cfd.nn.NsOneScaleGNN(checkpoint='NsOneScaleGNN.chk')`
- 2S-GNN: `NsTwoScaleGNN.chk`. Load with: `graphs4cfd.nn.NsTwoScaleGNN.load_model(checkpoint='NsTwoScaleGNN.chk')`
- 3S-GNN: `NsThreeScaleGNN.chk`. Load with: `graphs4cfd.nn.NsThreeScaleGNN(checkpoint='NsThreeScaleGNN.chk')`
- 4S-GNN: `NsFourScaleGNN.chk`. Load with: `graphs4cfd.nn.NsFourScaleGNN(checkpoint='NsFourScaleGNN.chk')`