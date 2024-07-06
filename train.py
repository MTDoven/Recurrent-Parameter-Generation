from model import BiARTransformer, DiffusionLoss
from dataset import Cifar10_MLP


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

import os


config = {
    # device setting
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # dataset setting
    "dataset": Cifar10_MLP,
    "data_path": "./dataset/cifar10_MLP/checkpoint",
    "dim_per_token": 2048,
    "sequence_length": 10260+1,
    "max_length": 1024,
    # model config
    "stage1_layers": 6,
    "stage2_layers": 6,
    "predict_length": 64,
    "num_heads": 8,
    "feedforward_dim": 2048,
    "dropout": 0.1,
    "transformer_activation": nn.GELU(),
    # diffusion loss config
    "mlp_layer_dims": [2048, 2048, 2048, 2048],
    "condition_dim": 2048,
    "diffusion_beta_max": 0.999,
    "diffusion_n_timesteps": 1000,
    "mlp_activation": nn.SiLU(),
    # train setting
    "batch_size": 1,
    "num_workers": 0,
    "epochs": 5000,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
}


# Data
print('==> Preparing data..')
train_set = config["dataset"](checkpoint_path=config["data_path"],
                              dim_per_token=config["dim_per_token"])
train_loader = DataLoader(dataset=train_set,
                          batch_size=config["batch_size"],
                          num_workers=config["num_workers"],
                          persistent_workers=False,
                          drop_last=True,
                          shuffle=True,)
def preprocess_data(datas):
    batch_size = datas.size(0)
    # start padding
    start_padding = model.start_padding.repeat(batch_size, 1, 1)
    datas = torch.cat([start_padding, datas], dim=1)
    # random cutting
    random_cutting = random.randint(1, datas.size(1) - config["predict_length"] - 1)
    inputs = datas[:, max(0, random_cutting-config["max_length"]):random_cutting, :]
    targets = datas[:, random_cutting:random_cutting + config["predict_length"], :]
    inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
    return inputs, targets

# Model
print('==> Building model..')
model = BiARTransformer(hidden_dim=config["dim_per_token"],
                        stage1_layers=config["stage1_layers"],
                        stage2_layers=config["stage2_layers"],
                        predict_length=config["predict_length"],
                        num_heads=config["num_heads"],
                        feedforward_dim=config["feedforward_dim"],
                        dropout=config["dropout"],
                        activation=config["transformer_activation"],)
model = model.to(config["device"])

# Loss
print('==> Building diffusion_loss..')
criterion = DiffusionLoss(mlp_layer_dims=config["mlp_layer_dims"],
                          condition_dim=config["condition_dim"],
                          diffusion_beta_max=config["diffusion_beta_max"],
                          diffusion_n_timesteps=config["diffusion_n_timesteps"],
                          device=config["device"],
                          mlp_activation=config["mlp_activation"],)
criterion = criterion.to(config["device"])

# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW(params=[{"params": model.parameters()},
                                {"params": criterion.parameters()}],
                        lr=config["learning_rate"],
                        weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                 T_max=config["epochs"])


# Training
print('==> Defining training..')

def train(epoch):
    print(f"Epoch: {epoch}", end=": ")
    model.train()
    train_loss = 0
    total = len(train_set)
    for batch_idx, datas in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = preprocess_data(datas)
        # train
        z = model(inputs)
        loss = criterion(targets, z)
        loss.backward()
        optimizer.step()
        # to logging losses
        train_loss += loss.item()
    print('Loss: %.6f' % (train_loss/total))

# def generate(save_name):
#     print("\n==> Testing..")
#     model.eval()
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
#             outputs = model(inputs.flatten(start_dim=1))
#             loss = criterion(outputs, targets)
#             # to logging losses
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#         print('\r', batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
#               (test_loss/(batch_idx+1), 100.*correct/total, correct, total), end="")
#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc and save_name is not None:
#         print('Saving..')
#         state = model.state_dict()
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, f'./checkpoint/{save_name}.pth')
#         best_acc = acc


best_acc = 0  # best test accuracy
if __name__ == '__main__':
    for epoch in range(0, config["epochs"]):
        epoch += 1
        train(epoch)
        scheduler.step()