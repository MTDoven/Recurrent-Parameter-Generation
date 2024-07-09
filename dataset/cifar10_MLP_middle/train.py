import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
try:  # relative import
    from .model import simple_mlp_for_cifar10_classify
except:  # absolute import
    from model import simple_mlp_for_cifar10_classify
import sys
import os


config = {
    # dataset setting
    "dataset_root": "/home/wangkai/AR-Param-Generation/Datasets",
    "classes": ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    # train setting
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 1000,
    "num_workers": 8,
    "learning_rate": 0.01,
    "epochs": 120,
    "test_freq": 10,
    "weight_decay": 1e-3,
}


# Data
print('==> Preparing data..')

train_loader = DataLoader(
    dataset=CIFAR10(
        root=config["dataset_root"],
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
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
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
)


# Model
print('==> Building model..')

model = simple_mlp_for_cifar10_classify().to(config["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),
                       lr=config["learning_rate"],
                       weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=config["epochs"])


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
        outputs = model(inputs.flatten(start_dim=1))
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
            outputs = model(inputs.flatten(start_dim=1))
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
            state[key] = value.cpu().to(torch.float16)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{save_name}')
        best_acc = acc


best_acc = 0  # best test accuracy
if __name__ == '__main__':
    # config save name
    items = os.listdir("./checkpoint")
    save_name = "0000.pth"
    while save_name in items:
        save_name = str(random.randint(0, 9999)).zfill(4) + '.pth'
    # main train
    for epoch in range(0, config["epochs"]):
        epoch += 1
        train(epoch, save_name)
        if epoch % config["test_freq"] == 0:
            test(save_name)
        scheduler.step()

    # fix some bug caused by num_workers
    del train_loader
    exit(0)