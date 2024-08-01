import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")
USE_WANDB = True

# other
import math
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import autocast
# model
from model import MambaDiffusion as Model
from model.diffusion import DDPMSampler, DDIMSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
# dataset
from dataset import Cifar10_ResNet18 as Dataset
from torch.utils.data import DataLoader



config = {
    # device setting
    "device": "cuda",
    # dataset setting
    "dataset": Dataset,
    "dim_per_token": 8192,
    "sequence_length": 1510,
    # train setting
    "batch_size": 4,
    "num_workers": 4,
    "total_steps": 80000,
    "learning_rate": 0.00003,
    "weight_decay": 0.0,
    "save_every": 80000//25,
    "print_every": 50,
    "warmup_steps": 1000,
    "autocast": lambda i: 10000 < i < 70000,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # to log
    "model_config": {
        # mamba config
        "d_condition": 1,
        "d_model": 8192,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        # diffusion config
        "diffusion_batch": 512,
        "layer_channels": [1, 64, 96, 64, 1],
        "model_dim": 8192,
        "condition_dim": 8192,
        "kernel_size": 7,
        "sample_mode": DDIMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        "forward_once": True,
    },
}


# Data
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=config["dim_per_token"])
print("Dataset length:", train_set.real_length)
print("input shape:", train_set[0].shape)
assert train_set.sequence_length == config["sequence_length"], f"sequence_length={train_set.sequence_length}"
train_loader = DataLoader(dataset=train_set,
                          batch_size=config["batch_size"],
                          num_workers=config["num_workers"],
                          persistent_workers=True,
                          drop_last=True,
                          shuffle=True,)

# Model
print('==> Building model..')
Model.config = config["model_config"]
model = Model(sequence_length=config["sequence_length"],
              device=config["device"])  # model setting is in model
model = model.to(config["device"])


# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW(params=model.parameters(),
                        lr=config["learning_rate"],
                        weight_decay=config["weight_decay"])
scheduler = SequentialLR(optimizer=optimizer,
                         schedulers=[LinearLR(optimizer=optimizer,
                                              start_factor=1e-4,
                                              end_factor=1.0,
                                              total_iters=config["warmup_steps"]),
                                     CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=config["total_steps"]-config["warmup_steps"])],
                         milestones=[config["warmup_steps"]],)

# wandb
if USE_WANDB:
    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(project="AR-Param-Generation", name=__file__.split("/")[-1][:-3], config=config,)




# Training
print('==> Defining training..')
def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> start training..")
    model.train()
    for batch_idx, param in enumerate(train_loader):
        optimizer.zero_grad()
        param = param.to(config["device"])
        # train
        with autocast(enabled=config["autocast"](batch_idx), dtype=torch.bfloat16):
            loss = model(param.shape, param)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # to logging losses and print and save
        if USE_WANDB:
            wandb.log({"train_loss": loss.item()})
        else:  # not use wandb
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx % config["save_every"] == 0:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
            torch.save(state, os.path.join(config["checkpoint_save_path"],
                                           f"{__file__.split('/')[-1].split('.')[0]}.pth"))
            generate(save_path=config["generated_path"], need_test=True)
        if batch_idx >= config["total_steps"]:
            break


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    # _, condition = train_set[0]
    with torch.no_grad():
        prediction = model.sample()  # condition=condition)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    if USE_WANDB:
        wandb.log({"generated_norm": generated_norm.item()})
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(config["test_command"])
        print("\n")
    model.train()
    return prediction


if __name__ == '__main__':
    train()
    del train_loader  # deal problems by dataloader
    print("Finished Training!")
    exit(0)