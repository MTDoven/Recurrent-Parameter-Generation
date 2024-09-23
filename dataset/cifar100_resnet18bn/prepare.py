# set global seed
import random
import numpy as np
import torch
seed = SEED = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)


try:  # relative import
    from model import Model
except ImportError:
    from .model import Model

# import
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100 as Dataset
from tqdm.auto import tqdm
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# load additional config
import json
config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_file, "r") as f:
    additional_config = json.load(f)




# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "dataset_root": "from_additional_config",
    "batch_size": 500 if __name__ == "__main__" else 200,
    "num_workers": 32,
    "learning_rate": 0.0005,
    "weight_decay": 0.000005,
    "epochs": 200,
    "save_learning_rate": 0.0,
    "total_save_number": 1,
    "tag": os.path.basename(os.path.dirname(__file__)),
}
config.update(additional_config)




# Data
dataset = Dataset(
    root=config["dataset_root"],
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.Resize(80),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy("cifar10")),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=Dataset(
        root=config["dataset_root"],
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.Resize(80),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])),
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    shuffle=False,
    pin_memory=True,
)

# Model
model, head = Model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
pre_optimizer = optim.AdamW(
    head.parameters(),
    lr=0.001,
    weight_decay=config["weight_decay"],
)
optimizer = optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"],
    eta_min=config["save_learning_rate"],
)




# Training
def train(model=model, optimizer=optimizer, scheduler=scheduler):
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),
                                             total=len(dataset) // config["batch_size"]):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()

# test
@torch.no_grad()
def test(model=model):
    model.eval()
    all_targets = []
    all_predicts = []
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader),
                                             total=len(test_loader.dataset) // config["batch_size"]):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        # to logging losses
        all_targets.extend(targets.flatten().tolist())
        test_loss += loss.item()
        _, predicts = outputs.max(1)
        all_predicts.extend(predicts.flatten().tolist())
        total += targets.size(0)
        correct += predicts.eq(targets).sum().item()
    loss = test_loss / (batch_idx + 1)
    acc = correct / total
    print(f"Loss: {loss:.4f} | Acc: {acc:.4f}")
    model.train()
    return loss, acc, all_targets, all_predicts

# save train
def save_train(model=model, optimizer=optimizer):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    # Save checkpoint
    _, acc, _, _ = test(model=model)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_state = {key: value.cpu().to(torch.float32) for key, value in model.state_dict().items()}
    torch.save(save_state, f"full_model.pth")
    print("save:", f"full_model.pth")




# main
if __name__ == '__main__':
    test(model=model)
    train(model=model, optimizer=pre_optimizer, scheduler=scheduler)
    train(model=model, optimizer=pre_optimizer, scheduler=scheduler)
    for epoch in range(config["epochs"]):
        train(model=model, optimizer=optimizer, scheduler=scheduler)
        test(model=model)
    save_train(model=model, optimizer=optimizer)