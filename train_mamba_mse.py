import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn import functional as F

from torch.utils.data import DataLoader
from model import MambaModel, ConditionalMLP, DiffusionLoss
from dataset import Cifar10_MLP, ImageDebugDataset

import wandb
import random
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


config = {
    # device setting
    "device": "cuda:7",
    # dataset setting
    "dataset": ImageDebugDataset,
    "data_path": "./dataset/image_debug_2m/checkpoint",
    "dim_per_token": 512,
    "sequence_length": 4050,
    "max_input_length": 64,
    # model config
    "dt_rank": 32,
    "dim_inner": 512,
    "d_state": 256,
    "dropout": 0.0,
    "depth": 6,
    # train setting
    "batch_size": 32,
    "num_workers": 8,
    "total_steps": 10000,
    "learning_rate": 0.00002,
    "weight_decay": 0.0,
    "save_every": 500,
    "print_every": 20,
    "warmup_steps": 250,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": "./dataset/image_debug_2m/generated/generated_classifier.pth",
    "test_command": "echo 'generated generated_classifier.pth'",
    # "test_command": "CUDA_VISIBLE_DEVICE=5 python " + \
    #                 "./dataset/cifar10_MLP_1M/test.py " + \
    #                 "./dataset/cifar10_MLP_1M/generated/generated_classifier.pth",
}


# wandb
wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
wandb.init(project="cifar10_MLP", config=config)

# Data
print('==> Preparing data..')
train_set = config["dataset"](checkpoint_path=config["data_path"],
                              dim_per_token=config["dim_per_token"],)
train_set.set_infinite_dataset()
print("Dataset length:", train_set.real_length)
print("Sequence length:", train_set[0].shape)
train_loader = DataLoader(dataset=train_set,
                          batch_size=config["batch_size"],
                          num_workers=config["num_workers"],
                          persistent_workers=True,
                          drop_last=True,
                          shuffle=True,)
def preprocess_data(datas):
    max_input_length = config["max_input_length"]
    sequence_length = config["sequence_length"]
    predict_length = 1
    assert max_input_length % predict_length == 0
    assert sequence_length % predict_length == 0
    random_cutting = random.randint(0, sequence_length // predict_length - 1) * predict_length
    inputs = datas[:, max(random_cutting-max_input_length, 0):random_cutting, :]
    targets = datas[:, random_cutting:random_cutting+predict_length, :]
    inputs, targets = inputs.to(config["device"], torch.float32), targets.to(config["device"], torch.float32)
    return inputs, targets

# Model
print('==> Building model..')
model = MambaModel(dim=config["dim_per_token"],  # Dimension of the model
                   dt_rank=config["dt_rank"],  # Rank of the dynamic routing matrix
                   dim_inner=config["dim_inner"],  # Inner dimension of the model
                   d_state=config["d_state"],  # Dimension of the state vector
                   dropout=config["dropout"],  # Dropout rate
                   depth=config["depth"],)  # Depth of the model
model = model.to(config["device"])

# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW(params=model.parameters(),
                        lr=config["learning_rate"],
                        weight_decay=config["weight_decay"],)
scheduler = SequentialLR(optimizer=optimizer,
                         schedulers=[LinearLR(optimizer=optimizer,
                                              start_factor=1e-4,
                                              end_factor=1.0,
                                              total_iters=config["warmup_steps"]),
                                     CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=config["total_steps"]-config["warmup_steps"])],
                         milestones=[config["warmup_steps"]],)


# Training
print('==> Defining training..')
total_steps = 0
train_loss = 0
this_steps = 0
def train():
    global total_steps, train_loss, this_steps
    model.train()
    for batch_idx, datas in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = preprocess_data(datas)
        # train
        prediction = model(inputs)
        loss = F.mse_loss(prediction, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # to logging losses and print and save
        train_loss += loss.item()
        wandb.log({"train_loss": loss.item(),
                   "z_norm": prediction.abs().mean(),})
        this_steps += 1
        total_steps += 1
        if this_steps % config["print_every"] == 0:
            print('Loss: %.6f' % (train_loss/this_steps))
            this_steps = 0
            train_loss = 0
        if total_steps % config["save_every"] == 0:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
            torch.save(state, os.path.join(config["checkpoint_save_path"], "state.pth"))
            generate(save_path=config["generated_path"], need_test=True)
        if total_steps >= config["total_steps"]:
            break


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        x = torch.zeros(size=(1, 0, config["dim_per_token"]), device=config["device"])
        prediction_list = []
        while True:
            this_prediction = model(x)
            prediction_list.append(this_prediction.cpu())
            x = torch.cat((x, this_prediction), dim=1)
            x = x[:, -config["max_input_length"]:, :]
            if len(prediction_list) == config["sequence_length"]:
                break
    prediction = torch.cat(prediction_list, dim=1)
    print("Generated_norm:", prediction.abs().mean())
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(config["test_command"])
        print("\n")
    model.train()
    return prediction


if __name__ == '__main__':
    # model = torch.compile(model, mode="default")
    train()

    # deal problems by dataloder
    del train_loader
    print("Finished Training!")
    exit(0)

