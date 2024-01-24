'''
    Script for training the NsThreeScaleGNN model on the NsCircle dataset.
    This model is referred to as the 3S-GNN in Lino et al. (2022) https://doi.org/10.1063/5.0097679.
'''


import torch
from torchvision import transforms
import graphs4cfd as gfd


# Training configuration
train_config = gfd.nn.TrainConfig(
    name            = 'NsThreeScaleGNN',
    folder          = '.',
    tensor_board    = '.',
    chk_interval    = 1,
    training_loss   = gfd.nn.losses.GraphLoss(lambda_d=0.25),
    validation_loss = gfd.nn.losses.GraphLoss(),
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
    gfd.transforms.ConnectKNN(6, period=[None, "auto"]),
    gfd.transforms.ScaleNs({'u': (-2.1,2.6), 'v': (-2.25,2.1), 'p': (-3.7,2.35), 'Re': (500,1000)}, format='uvp'),
    gfd.transforms.ScaleEdgeAttr(0.1),
    gfd.transforms.RandomGraphRotation(eq='ns', format='uvp'),
    gfd.transforms.RandomGraphFlip(eq='ns', format='uvp'),
    gfd.transforms.AddUniformNoise(0.01)
])
batch_transform = transforms.Compose([
    gfd.transforms.GridClustering([0.15, 0.30]),
])
dataset = gfd.datasets.NsCircle(format='uvp', path=path, training_info={"n_in":1, "n_out":train_config['num_steps'][-1], "step":1, "T":100}, transform=transform) # If enough memory, set preload=True
train_set, test_set = torch.utils.data.random_split(dataset, [1000,32])
train_loader = gfd.DataLoader(
    train_set,
    batch_size  = train_config['batch_size'],
    shuffle     = True,
    transform   = batch_transform,
    num_workers = 4
)   
val_loader  = gfd.DataLoader(
    test_set, 
    batch_size  = train_config['batch_size'],
    shuffle     = False,
    transform   = batch_transform,
    num_workers = 4
)   

# Model definition
arch = {
    ################ Edge-functions ################## Node-functions ##############
    # Encoder
    "edge_encoder": (2, (128,128,128), False),
    "node_encoder": (5, (128,128,128), False),
    # Level 1
    "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "down_mp12": (2+128, (128,128,128), True),
    # Level 2
    "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "down_mp23": (2+128, (128,128,128), True),
    # Level 3
    "mp31": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp32": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp33": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp34": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "up_mp32": (2+128+128, (128,128,128), True),
    # Level 2
    "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "up_mp21": (2+128+128, (128,128,128), True),
    # Level 1
    "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Decoder
    "decoder": (128, (128,128,3), False),
}  
model = gfd.nn.NsThreeScaleGNN(arch=arch)
print("Number of trainable parameters: ", model.num_params)


# Training
model.fit(train_config, train_loader, val_loader=val_loader)

