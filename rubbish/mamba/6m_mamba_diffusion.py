import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")

USE_WANDB = True
FINAL_RUNNING = True
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from model import MambaDiffusion
from model import DDIMSampler
from dataset.BaseDataset import Cifar10_GoogleNet
if USE_WANDB:
    import wandb
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


config = {
    # device setting
    "device": "cuda:4",
    # dataset setting
    "dataset": Cifar10_GoogleNet,
    "dim_per_token": 4096,
    "sequence_length": 6179,
    "max_input_length": 6179,
    # train setting
    "batch_size": 1,
    "num_workers": 2,
    "total_steps": 60000,
    "learning_rate": 0.0002,
    "weight_decay": 0.0,
    "save_every": 1000,
    "print_every": 50,
    "warmup_steps": 500,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Cifar10_GoogleNet.generated_path,
    "test_command": Cifar10_GoogleNet.test_command,
    # to log
    "model_config": {
        # mamba config
        "d_condition": 1,
        "d_model": 4096,
        "d_state": 32,
        "d_conv": 4,
        "expand": 2,
        # diffusion config
        "diffusion_batch": 512,
        "layer_channels": [1, 32, 64, 96, 64, 32, 1],
        "model_dim": 1024,
        "condition_dim": 4096,
        "kernel_size": 5,
        "sample_mode": DDIMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        "forward_once": True,
    },
}


# Data
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=config["dim_per_token"],
                              max_input_length=config["max_input_length"],)
train_set.set_infinite_dataset().set_return_full_param()
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
MambaDiffusion.config = config["model_config"]
model = MambaDiffusion(sequence_length=config["sequence_length"],
                       device=config["device"])  # model setting is in model
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

# wandb
if USE_WANDB:
    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(project="cifar10_MLP_final" if FINAL_RUNNING else "cifar10_MLP",
               name=__file__.split("/")[-1],
               config=config,)




# Training
print('==> Defining training..')
total_steps = 0
train_loss = 0
this_steps = 0
def train():
    global total_steps, train_loss, this_steps
    model.train()
    for batch_idx, param in enumerate(train_loader):
        optimizer.zero_grad()
        param = param.to(config["device"])
        # train
        # with torch.cuda.amp.autocast(enabled=batch_idx < config["total_steps"] * 0.75, dtype=torch.bfloat16):
        loss = model(param.shape, param)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # to logging losses and print and save
        train_loss += loss.item()
        if USE_WANDB:
            wandb.log({"train_loss": loss.item()})
        this_steps += 1
        total_steps += 1
        if this_steps % config["print_every"] == 0:
            if not FINAL_RUNNING:
                print('Loss: %.6f' % (train_loss/this_steps))
            this_steps = 0
            train_loss = 0
        if total_steps % config["save_every"] == 0:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
            torch.save(state, os.path.join(config["checkpoint_save_path"],
                                           f"{__file__.split('/')[-1].split('.')[0]}.pth"))
            generate(save_path=config["generated_path"], need_test=True)
        if total_steps >= config["total_steps"]:
            break


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        prediction = model.sample()
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
    # model = torch.compile(model, mode="default")
    train()

    # deal problems by dataloder
    del train_loader
    print("Finished Training!")
    exit(0)

