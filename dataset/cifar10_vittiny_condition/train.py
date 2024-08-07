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
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast
try:  # relative import
    from .model import imagenet_classify as Model
    from .dataset import OneClassCIFAR10
except:  # absolute import
    from model import imagenet_classify as Model
    from dataset import OneClassCIFAR10
# other
from tqdm.auto import tqdm
import json
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
config_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Cifar10")
with open(os.path.join(config_root, "config.json"), "r") as f:
    additional_config = json.load(f)


config = {
    # dataset setting
    "dataset_root": "from_additional_config",
    "optim_class": int(sys.argv[1]),
    # train setting
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 250,
    "num_workers": 20,
    "pre_learning_rate": 0.001,
    "pre_epoch": 3,
    "learning_rate": 0.00001,
    "epochs": 55,
    "weight_decay": 0.1,
    "autocast": True,
    "start_save": 15,
    "debug_iteration": 100,
    "tag": os.path.dirname(__file__).split("_")[-2],
}
config.update(additional_config)
print(f"start training optim_class:{config['optim_class']}")




# Data
print('==> Preparing data..')

train_loader = DataLoader(
    dataset=OneClassCIFAR10(
        dataset_root=config["dataset_root"],
        optim_class=config["optim_class"],
        train=True,
    ),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
)
test_loader = DataLoader(
    dataset=OneClassCIFAR10(
        dataset_root=config["dataset_root"],
        optim_class=config["optim_class"],
        train=False,
    ),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
)


# Model
print('==> Building model..')

model = Model()
model = model.to(config["device"])
criterion = nn.BCEWithLogitsLoss()

pre_optimizer = optim.AdamW(
    model.head.parameters(),
    lr=config["pre_learning_rate"],
    weight_decay=config["weight_decay"],
)
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

def pre_train():
    model.train()
    for e in range(config["pre_epoch"]):
        for batch_idx, (inputs, targets) in tqdm(
                enumerate(train_loader),
                total=len(train_loader.dataset) // config["batch_size"]):
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
            pre_optimizer.zero_grad()
            with autocast(enabled=config["autocast"], dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            pre_optimizer.step()

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
        with autocast(enabled=config["autocast"], dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # to logging losses
        train_loss += loss.item()
        predicted = torch.where(outputs > 0., 1, 0)
        correct += predicted.eq(targets).sum().item()
        total += len(outputs)
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
            predicted = torch.where(outputs > 0., 1, 0)
            correct += predicted.eq(targets).sum().item()
            total += len(outputs)
        print('\r', batch_idx, len(test_loader), 'Loss: %.4f | Acc: %.4f%% (%d/%d)' %
              (test_loss/(batch_idx+1), 100.*correct/total, correct, total), end="")
    # Save checkpoint.
    if save_name is not None and int(save_name) >= config["start_save"]:
        print('\tSaving..')
        state = {}
        for key, value in model.state_dict().items():
            state[key] = value.cpu().to(torch.float32)
        os.makedirs('checkpoint', exist_ok=True)
        torch.save(state, f"checkpoint/{save_name}_class{config['optim_class']}_acc{correct/total:.4f}_{config['tag']}.pth")




best_acc = 0  # best test accuracy
if __name__ == '__main__':
    # config save name
    save_name = 0

    # main train
    pre_train()
    test(None)
    for epoch in range(0, config["epochs"]):
        epoch += 1
        train(epoch, str(save_name).zfill(4))
        test(str(save_name).zfill(4))
        scheduler.step()
        save_name += 1

    # fix some bug caused by num_workers
    del train_loader
    exit(0)
