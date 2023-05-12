import torch
from torchvision import transforms
import graphs4cfd as gcfd


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on",device)

# Training configuration
train_config = gcfd.nn.TrainConfig(
    name            = 'NSL4',
    folder          = '.',
    tensor_board    = '.',
    chk_interval    = 1,
    training_loss   = gcfd.nn.losses.GraphLoss(lambda_d=0.25),
    validation_loss = gcfd.nn.losses.GraphLoss(),
    epochs          = 500,
    num_steps       = [i for i in range(1,11)],
    add_steps       = {'tolerance': 0.005, 'loss': 'training'},
    batch_size      = 8,
    lr              = 1e-5,
    grad_clip       = {"epoch": 0, "limit": 1},
    scheduler       = {"factor": 0.5, "patience": 5, "loss": 'training'},
    stopping        = 1e-8,
)

# Datasets
path = "/rds/general/user/mal1218/ephemeral/NS/nsCircle_short.h5"
transform = transforms.Compose([
    gcfd.transforms.ConnectKNN(6, period=[None, "auto"]),
    gcfd.transforms.ScaleNs({'u': (-2.1,2.6), 'v': (-2.25,2.1), 'p': (-3.7,2.35), 'Re': (500,1000)}),
    gcfd.transforms.ScaleEdgeAttr(0.1),
    gcfd.transforms.RandomGraphRotation(eq='ns', format='uvp'),
    gcfd.transforms.RandomGraphFlip(eq='ns', format='uvp'),
    gcfd.transforms.AddUniformNoise(0.01)
])
dataset = gcfd.datasets.NsCircle(format='uvp', path=path, training_info={"n_in":1, "n_out":train_config['num_steps'][-1], "step":1, "T":100}, transform=transform, preload=False)
train_set, test_set = torch.utils.data.random_split(dataset, [1000,32])
train_loader = gcfd.DataLoader(train_set, batch_size=train_config['batch_size'], shuffle=True,  num_workers=4)   
test_loader  = gcfd.DataLoader(test_set,  batch_size=train_config['batch_size'], shuffle=False, num_workers=4)   

# Model
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
    "pool12": ((2+128, (128,128,128), True), 0.15),
    # Level 2
    "mp211": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp212": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "pool23": ((2+128, (128,128,128), True), 0.30),
    # Level 3
    "mp311": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp312": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "pool34": ((2+128, (128,128,128), True), 0.60),
    # Level 4
    "mp41": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp42": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp43": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp44": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "unpool43": ((2+128+128, (128,128,128), True), 0.60),
    # Level 3
    "mp321": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp322": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "unpool32": ((2+128+128, (128,128,128), True), 0.30),
    # Level 2
    "mp221": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp222": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "unpool21": ((2+128+128, (128,128,128), True), 0.15),
    # Level 1
    "mp121": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp122": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp123": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    "mp124": ((128+2*128, (128,128,128), True), (128+128, (128,128,128), True)),
    # Decoder
    "decoder": (128, (128,128,3), False),
}
model = gcfd.nn.NsFourScaleGNN(arch=arch, device=device)
print("Number of trainable parameters: ", model.num_params)

# Train
model.fit(train_config, train_loader, test_loader=test_loader)

