'''
    Script for training the NsTwoGuillardScaleGNN model on the NsCircle dataset.
    This model is the two-scale GNN in Appendix C.3 in Lino et al. (2022) https://doi.org/10.1063/5.0097679.
'''


import torch
from torchvision import transforms
import graphs4cfd as gfd


# Training configuration
config = gfd.nn.TrainConfig(
    name            = 'NsTwoGuillardScaleGNN',
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
    gfd.transforms.GuillardCoarseningAndConnectKNN(k=(6,6), period=(None,"auto"), scale_edge_attr=(0.1, 0.25)),
    gfd.transforms.ScaleNs({"u": (-2.1,2.6), "v": (-2.25,2.1), "p": (-3.7,2.35), "Re": (500,1000)}, format='uvp'),
    gfd.transforms.BuildKnnInterpWeights(6),
    gfd.transforms.RandomGraphRotation(eq='ns', format='uvp'),
    gfd.transforms.RandomGraphFlip(eq='ns', format='uvp'),
    gfd.transforms.AddUniformNoise(0.01),
])
batchTransform = transforms.Compose([
     gfd.transforms.BuildKnnInterpWeights(6),
]) 
dataset = gfd.datasets.NsCircle(format='uvp', path=path, training_info={"n_in":1, "n_out":config['num_steps'][-1], "step":1, "T":100}, transform=transform) # If enough memory, set preload=True
train_set, test_set = torch.utils.data.random_split(dataset, [1000,32])
train_loader = gfd.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True , transform=batchTransform, num_workers=4)   
test_loader  = gfd.DataLoader(test_set,  batch_size=config['batch_size'], shuffle=False, transform=batchTransform, num_workers=4)   


# Model definition
arch = {
    ################ Edge-functions ################## Node-functions ##############
    # Encoder
    "edge_encoder" : (2, (128,128,128), False),
    "edge_encoder2": (2, (128,128,128), False),
    "node_encoder" : (5, (128,128,128), False),
    # Level 1
    "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp113": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp114": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 2
    "mp21": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp22": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp23": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp24": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Level 1
    "mp121": ((128+2*256, (128,128,128), True), (128+256, (128,128,128), True)),
    "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Decoder
    "decoder": (128, (128,128,3), False),
} 
model = gfd.nn.NsTwoGuillardScaleGNN(arch=arch)


# Training
model.fit(config, train_loader, val_loader=val_loader)

