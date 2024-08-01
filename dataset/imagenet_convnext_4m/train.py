import random
import numpy as np
import torch
seed = SEED = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.cuda.amp import autocast
try:  # relative import
    from .model import imagenet_classify as Model
    from .dataset import ImageNet1k as Dataset
except:  # absolute import
    from model import imagenet_classify as Model
    from dataset import ImageNet1k as Dataset

from tqdm import tqdm
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


config = {
    # dataset setting
    "train_image_root": "/home/wangkai/datasets/imagenet/train",
    "test_image_root": "/home/wangkai/datasets/imagenet/val",
    "train_mapping_dict": "/home/wangkai/datasets/imagenet/train_mapping.dict",
    "test_mapping_dict": "/home/wangkai/datasets/imagenet/val_mapping.dict",
    # train setting
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 64,
    "num_workers": 16,
    "learning_rate": 0.00001,
    "epochs": 10,
    "start_save_ratio": 0.2,
    "save_every": 1200,
    "weight_decay": 0.0001,
    "autocast": True,
    "debug_iteration": sys.maxsize,
}


# Data
print('==> Preparing data..')

train_loader = DataLoader(
    dataset=Dataset(
        image_root=config["train_image_root"],
        mapping_dict=config["train_mapping_dict"],
    ),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    dataset=Dataset(
        image_root=config["test_image_root"],
        mapping_dict=config["test_mapping_dict"],
    ),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
)


# Model
print('==> Building model..')

model = Model()
model = model.to(config["device"])
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"],
    eta_min=1e-8,
)


# Training
print('==> Defining training..')
def train(epoch, save_name):
    print(f"\nEpoch: {epoch}", end=": ")
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(
            enumerate(train_loader),
            total=len(train_loader.dataset) // config["batch_size"]):
        inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
        optimizer.zero_grad()
        with autocast(enabled=config["autocast"] and \
                              epoch >= config["epochs"] * config["start_save_ratio"],
                      dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # to logging losses
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % config["save_every"] == 0 \
                and save_name is not None \
                and epoch >= config["epochs"] * config["start_save_ratio"]:
            print('\tSaving..')
            state = {}
            for key, value in model.state_dict().items():
                state[key] = value.cpu().to(torch.float32)
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(state, f'checkpoint/{save_name}_acc{correct / total:.4f}_seed{SEED}_tinyvit.pth')
        if batch_idx > config["debug_iteration"]:
            break
    print('\r', 'Loss: %.4f | Acc: %.4f%% (%d/%d)' %
          (train_loss / (batch_idx + 1), 100. * correct / total, correct, total), end="")


@torch.no_grad()
def test(save_name):
    print("\n==> Testing..")
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # to logging losses
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('\r', batch_idx, len(test_loader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)' %
              (test_loss/(batch_idx+1), 100.*correct/total, correct, total), end="")
    # Save checkpoint.
    if save_name is not None:
        print('\tSaving..')
        state = {}
        for key, value in model.state_dict().items():
            state[key] = value.cpu().to(torch.float32)
        os.makedirs('checkpoint', exist_ok=True)
        torch.save(state, f'checkpoint/{save_name}_acc{correct / total:.4f}_seed{SEED}_tinyvit.pth')




best_acc = 0  # best test accuracy
if __name__ == '__main__':
    # config save name
    save_name = 0

    # main train
    for epoch in range(0, config["epochs"]):
        epoch += 1
        train(epoch, str(save_name).zfill(4))
        test(str(save_name).zfill(4))
        scheduler.step()
        save_name += 1

    # fix some bug caused by num_workers
    del train_loader
    exit(0)