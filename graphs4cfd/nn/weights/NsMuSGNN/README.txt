Paameters of the MuS-GNN models for simulating incompressible flow pass a vertical array of vertical cylinders from Lino et al. (2022) (https://doi.org/10.1063/5.0097679).

The models included are:
    - 1S-GNN: NsOneScaleGNN.chk. Load with: graphs4cfd.nn.mus_gnn.NsOneScaleGNN.load_model('NsOneScaleGNN.chk')
    - 2S-GNN: NsTwoScaleGNN.chk. Load with: graphs4cfd.nn.mus_gnn.NsTwoScaleGNN.load_model('NsTwoScaleGNN.chk')
    - 3S-GNN: NsThreeScaleGNN.chk. Load with: graphs4cfd.nn.mus_gnn.NsThreeScaleGNN.load_model('NsThreeScaleGNN.chk')
    - 4S-GNN: NsFourScaleGNN.chk. Load with: graphs4cfd.nn.mus_gnn.NsFourScaleGNN.load_model('NsFourScaleGNN.chk')