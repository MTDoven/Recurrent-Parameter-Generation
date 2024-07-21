import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
try:  # relative import
    from .model import mobile_net_for_cifar10_classify as Model
except:  # absolute import
    from model import mobile_net_for_cifar10_classify as Model
import sys
import os


config = {
    # dataset setting
    "dataset_root": "/home/wangkai/AR-Param-Generation/Datasets",
    "classes": ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    # train setting
    "device": "cuda",
    "batch_size": 256,
    "num_workers": 24,
    "learning_rate": 0.005,
    "epochs": 150,
    "test_freq": 10,
    "weight_decay": 0.005,
}


# Data
print('==> Preparing data..')

train_loader = DataLoader(
    dataset=CIFAR10(
        root=config["dataset_root"],
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    dataset=CIFAR10(
        root=config["dataset_root"],
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
)


# Model
print('==> Building model..')

model = Model().to(config["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),
                       lr=config["learning_rate"],
                       weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=config["epochs"],
                                                 eta_min=1e-5,)


# Training
print('==> Defining training..')

def train(epoch, save_name):
    print(f"\nEpoch: {epoch}", end=": ")
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # to logging losses
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('\r', 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
          (train_loss/(batch_idx+1), 100.*correct/total, correct, total), end="")
    if epoch >= config["epochs"] * 1 / 3:
        state = {}
        for key, value in model.state_dict().items():
            state[key] = value.cpu().to(torch.float32)
        torch.save(state, f'./checkpoint/{save_name}.pth')

def test(save_name):
    print("\n==> Testing..")
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # to logging losses
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('\r', batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
              (test_loss/(batch_idx+1), 100.*correct/total, correct, total), end="")
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc and save_name is not None:
        print('\tSaving..')
        state = {}
        for key, value in model.state_dict().items():
            state[key] = value.cpu().to(torch.float32)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{save_name}.pth')
        best_acc = acc


best_acc = 0  # best test accuracy
if __name__ == '__main__':
    # config save name
    save_name = 0
    # main train
    for epoch in range(0, config["epochs"]):
        epoch += 1
        train(epoch, str(save_name).zfill(4))
        if epoch % config["test_freq"] == 0:
            test(str(save_name).zfill(4))
        scheduler.step()
        save_name += 1

    # fix some bug caused by num_workers
    del train_loader
    exit(0)