'''
    Script for training the NsFourGuillardScaleGNN model on the NsCircle dataset.
    This model is the four-scale GNN in Appendix C.3 in Lino et al. (2022) https://doi.org/10.1063/5.0097679.
'''


import torch
from torchvision import transforms
import graphs4cfd as gcfd


# Training configuration
config = gcfd.models.Config(
    name            = 'NsFourGuillardScaleGNN',
    folder          = '.',
    tensor_board    = '.',
    chk_interval    = 1,
    training_loss   = gcfd.losses.GraphLoss(lambda_d=0.25),
    validation_loss = gcfd.losses.GraphLoss(),
    epochs          = 500,
    num_steps       = [i for i in range(1,11)],
    add_steps       = {'tolerance': 0.005, 'loss': 'training'},
    batch_size      = 8,
    lr              = 1e-5,
    grad_clip       = {"epoch": 0, "limit": 1},
    scheduler       = {"factor": 0.5, "patience": 5, "loss": 'training'},
    stopping        = 1e-8,
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)


# Training datasets
path = "<Path to dataset NsCircle.h5>" # Replace with path to NsCircle.h5 (available at https://doi.org/10.5281/zenodo.7870707)
transform = transforms.Compose([
    gcfd.transforms.GuillardCoarseningAndConnectKNN(k=(6,6,6,6), period=(None,"auto"), scale_edge_attr=(0.1, 0.25, 0.5, 1)),
    gcfd.transforms.ScaleNs({"u": (-2.1,2.6), "v": (-2.25,2.1), "p": (-3.7,2.35), "Re": (500,1000)}, format='uvp'),
    gcfd.transforms.RandomGraphRotation(eq='ns'),
    gcfd.transforms.RandomGraphFlip(eq='ns'),
    gcfd.transforms.AddUniformNoise(0.01),
])
dataset = gcfd.datasets.NsCircle(path=path, training_info={"n_in":1, "n_out":config['num_steps'][-1], "step":1, "T":100}, transform=transform, preload=True) # If not enough memory, set preload=False
train_set, test_set = torch.utils.data.random_split(dataset, [1000,32])
train_loader = gcfd.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True , num_workers=4)   
test_loader  = gcfd.DataLoader(test_set,  batch_size=config['batch_size'], shuffle=False, num_workers=4)   


# Model definition
arch = {
    ################ Edge-functions ################## Node-functions ##############
    # Encoder
    "edge_encoder" : (2, (128,128,128), False),
    "edge_encoder2": (2, (128,128,128), False),
    "edge_encoder3": (2, (128,128,128), False),
    "edge_encoder4": (2, (128,128,128), False),
    "node_encoder" : (5, (128,128,128), False),
    # Level 1
    "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 2
    "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 3
    "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 4
    "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 3
    "mp321": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
    "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 2
    "mp221": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
    "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 1
    "mp121": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
    "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Decoder
    "decoder": (128, (128,128,3), False),
}
model = gcfd.nn.NsFourGuillardScaleGNN(arch=arch)


# Training
model.fit(config, train_loader, test_loader=test_loader)

