USE_WANDB = True
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from model.lstm import LstmDiffusion, LstmModel
from model.diffusion import DiffusionLoss
from dataset.Dataset import RandomDebugDataset
import os
if USE_WANDB:
    import wandb
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


config = {
    # device setting
    "device": "cuda:5",
    # dataset setting
    "dataset": RandomDebugDataset,
    "dim_per_token": 1024,
    "sequence_length": 10,
    "max_input_length": 10,
    # train setting
    "batch_size": 8,
    "num_workers": 2,
    "total_steps": 50000,
    "learning_rate": 0.0001,
    "weight_decay": 0.0,
    "save_every": 2000,
    "print_every": 50,
    "warmup_steps": 500,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": RandomDebugDataset.generated_path,
    "test_command": RandomDebugDataset.test_command,
    # to log
    "model_config": LstmModel.config,
    "diffusion_config": DiffusionLoss.config
}


# Data
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=config["dim_per_token"],
                              max_input_length=config["max_input_length"],
                              test_tensor=torch.randn((config["sequence_length"], config["dim_per_token"]),
                                                      requires_grad=False))
train_set.return_full_param = True
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
model = LstmDiffusion(sequence_length=config["sequence_length"],
                      device=config["device"])  # model setting is in model
model = model.to(config["device"])


# Optimizer
print('==> Building optimizer..')
optimizer = optim.RMSprop(params=model.parameters(),
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
    wandb.init(project="cifar10_MLP", config=config)




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
        loss = model(param.shape, param)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # to logging losses and print and save
        train_loss += loss.item()
        if USE_WANDB:
            wandb.log({"train_loss": loss.item(),})
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

