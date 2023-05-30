'''
    Script for training the AdvThreeScaleGNN model on the AdvBox and AdvInBox datasets.
    This model is referred to as the 3S-GNN in Lino et al. (2022) https://doi.org/10.1063/5.0097679.
'''


import torch
from torchvision import transforms
import graphs4cfd as gfd


# Training configuration
train_config = gfd.nn.TrainConfig(
    name            = 'AdvThreeScaleGNN',
    folder          = '.',
    tensor_board    = '.',
    chk_interval    = 1,
    training_loss   = gfd.nn.losses.GraphLoss(lambda_d=0.25),
    validation_loss = gfd.nn.losses.GraphLoss(),
    epochs          = 500,
    num_steps       = [i for i in range(1,11)],
    add_steps       = {'tolerance': 0.01, 'loss': 'training'},
    batch_size      = 8,
    lr              = 1e-4,
    grad_clip       = {"epoch": 0, "limit": 1},
    scheduler       = {"factor": 0.5, "patience": 5, "loss": 'training'},
    stopping        = 1e-8,
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)


# Training datasets
path1 = "<Path to AdvBox.h5>" # Replace with path to AdvBox.h5 (available at https://doi.org/10.5281/zenodo.7861710)
transform1 = transforms.Compose([
    gfd.transforms.InterpolateNodesToXml("<Path to folder AdvBox_nodes_xml>"), # Replace with path to folder /box_nodes_xml (available at https://doi.org/10.5281/zenodo.7944488
    gfd.transforms.ConnectKNN(6, period=(1, 1)),
    gfd.transforms.ScaleEdgeAttr(0.01),
    gfd.transforms.GridClustering([0.02, 0.04]),
    gfd.transforms.RandomGraphRotation(eq='adv'),
    gfd.transforms.RandomGraphFlip(eq='adv'),
    gfd.transforms.AddUniformNoise(0.01)
])
dataset1 = gfd.datasets.Adv(path=path1, training_info={"n_in":1, "n_out":10, "step":2, "T":100}, transform=transform1) # If enough memory, set preload=True

path2 = "<Path to AdvInBox.h5>" # Replace with path to AdvInlet.h5 (available at https://doi.org/10.5281/zenodo.7861710)
transform2 = transforms.Compose([
    gfd.transforms.InterpolateNodesToXml("<Path to AdvInBox_nodes_xml>"), # Replace with path to folder /inlet_nodes_xml (available at https://doi.org/10.5281/zenodo.7944488
    gfd.transforms.ConnectKNN(6, period=(None, 0.5)),
    gfd.transforms.ScaleEdgeAttr(0.01),
    gfd.transforms.GridClustering([0.02, 0.04]),
    gfd.transforms.RandomGraphRotation(eq='adv'),
    gfd.transforms.RandomGraphFlip(eq='adv'),
    gfd.transforms.AddUniformNoise(0.01)
])

dataset2 = gfd.datasets.Adv(path=path2, training_info={"n_in":1, "n_out":10, "step":2, "T":100}, transform=transform2) # If enough memory, set preload=True
train_set1,  test_set1 = torch.utils.data.random_split(dataset1, [1490,10])
train_set2,  test_set2 = torch.utils.data.random_split(dataset2, [2990,10])
train_set    = train_set1 + train_set2
test_set     = test_set1  + test_set2
train_loader = gfd.DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True,  num_workers=4)   
test_loader  = gfd.DataLoader(test_set,  batch_size=train_config['batch_size'], shuffle=False, num_workers=4)   


# Model definition
arch = {
    ################ Edge-functions ################## Node-functions ##############
    # Encoder
    "edge_encoder": (2, (128,128,128), False),
    "node_encoder": (4, (128,128,128), False),
    # Level 1
    "mp111": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp112": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "down_mp12": (2+128, (128,128,128), True),
    # Level 2
    "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "down_mp23": (2+128, (128,128,128), True),
    # Level 3
    "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "down_mp34": (2+128, (128,128,128), True),
    # Level 4
    "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "up_mp43": (2+128+128, (128,128,128), True),
    # Level 3
    "mp321": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "up_mp32": (2+128+128, (128,128,128), True),
    # Level 2
    "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "up_mp21": (2+128+128, (128,128,128), True),
    # Level 1
    "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Decoder
    "decoder": (128, (128,128,1), False),
}
model = gfd.nn.AdvThreeScaleGNN(arch=arch)


# Training
model.fit(train_config, train_loader, val_loader=test_loader)

