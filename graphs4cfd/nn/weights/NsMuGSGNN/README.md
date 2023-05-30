Parameters of the multi-scale GNNs with low-resolution graphs otained by Guillard's coarsening.
See Lino et al. (2022) ([https://doi.org/10.1063/5.0097679)](https://doi.org/10.1063/5.0097679)).

The models included are:
- `NsTwoGuillardScaleGNN.chk`. Load with: `graphs4cfd.nn.NsTwoScaleGNN(checkpoint='NsTwoGuillardScaleGNN.chk')`
- `NsThreeGuillardScaleGNN.chk`. Load with: `graphs4cfd.nn.NsThreeScaleGNN(checkpoint='NsThreeGuillardScaleGNN.chk')`
- `NsFourGuillardScaleGNN.chk`. Load with: `graphs4cfd.nn.NsFourScaleGNN(checkpoint='NsFourGuillardScaleGNN.chk')`