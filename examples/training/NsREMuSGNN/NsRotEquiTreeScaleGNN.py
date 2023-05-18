'''
    Script for training the NsRotEquivThreeScaleGNN model on the NsEllipse dataset.
    This model is the three-scale REMuS-GNN in Lino et al. (2022) https://doi.org/10.1063/5.0097679.
'''


import torch
from torchvision import transforms
import graphs4cfd as gcfd


# Training configuration
config = gcfd.models.Config(
    name            = 'NsRotEquiThreeScaleGNN',
    folder          = '.',
    tensor_board    = '.',
    chk_interval    = 1,
    training_loss   = gcfd.losses.GraphLoss(lambda_d=0.25),
    validation_loss = gcfd.losses.GraphLoss(),
    epochs          = 500,
    num_steps       = [i for i in range(1,11)],
    add_steps       = {'tolerance': 0.002, 'loss': 'training'},
    batch_size      = 4,
    lr              = 1e-5,
    grad_clip       = {"epoch": 0, "limit": 1},
    scheduler       = {"factor": 0.5, "patience": 1, "loss": 'training'},
    stopping        = 1e-8,
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)


# Training datasets
path = "<Path to dataset NsEllipse.h5>" # Replace with path to NsEllipse.h5 (available at https://doi.org/10.5281/zenodo.7892171)
transform = transforms.Compose([
    gcfd.transforms.RandomNodeSubset(0.8),
    gcfd.transforms.ScaleNs({'u': (-1.8,1.8), 'v': (-1.8,1.8), "Re": (500,1000)}, format='uv'),
    gcfd.transforms.BuildRemusGraph(num_levels=3, k=5, scale_edge_length=(0.1, 0.2, 0.4)),
    gcfd.transforms.AddUniformNoise(0.01),
])
train_set = gcfd.datasets.NsEllipse(path=path, training_info={"n_in":1, "n_out":10, "step":1, "T":101}, transform=transform, preload=True) # If not enough memory, set preload=False
batch_transform = transforms.Compose([
     gcfd.transforms.BuildKnnInterpWeights(5),
])
train_loader = gcfd.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, transform=batch_transform, num_workers=4)   


# Model definition
arch = {
    ################ Angle-functions ################## Edge-functions ##############
    # Encoder
    "angle_encoder"  : (4, (128,128), True),
    "angle_encoder12": (4, (128,128), True),
    "angle_encoder2" : (4, (128,128), True),
    "angle_encoder23": (4, (128,128), True),
    "angle_encoder3" : (4, (128,128), True),
    "edge_encoder"   : (3, (128,128), True),
    "edge_encoder2"  : (3, (128,128), True),
    "edge_encoder3"  : (3, (128,128), True),
    # Level 1
    "mp111": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp112": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp113": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp114": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    # Pooling 1->2
    "down_mp12": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    # Level 2
    "mp211": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp212": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    # Pooling 2->3
    "down_mp23": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    # Level 3
    "mp31": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp32": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp33": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp34": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    # Undown_mping 3->2
    "up_mp32": ((128+128, (128,128,128), True),), ###### WRONG
    # Level 2
    "mp221": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp222": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    # Undown_mping 2->1
    "up_mp21": ((128+128, (128,128,128), True),), ###### WRONG
    # Level 1
    "mp121": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp122": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp123": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    "mp124": ((128+2*128, (128,128), True), (128+128, (128,128), True)),
    # Decoder
    "decoder": (128, (128,1), False),
}
model = gcfd.nn.NsRotEquiTreeScaleGNN(arch=arch)


# Training
model.fit(config, train_loader)

