import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from model import BiARTransformer, DiffusionLoss
from dataset import Cifar10_MLP
import random
import os


config = {
    # device setting
    "device": "cuda:0",
    # dataset setting
    "dataset": Cifar10_MLP,
    "data_path": "./dataset/cifar10_MLP/checkpoint",
    "dim_per_token": 1024,
    "sequence_length": 10005,
    "max_length": 1024,
    # model config
    "stage1_layers": 6,
    "stage2_layers": 6,
    "predict_length": 64,
    "num_heads": 8,
    "feedforward_dim": 1024,
    "dropout": 0.1,
    "transformer_activation": nn.GELU(),
    # diffusion loss config
    "mlp_layer_dims": [1024, 1024, 1024, 1024],
    "condition_dim": 1024,
    "diffusion_beta_max": 0.999,
    "diffusion_n_timesteps": 1000,
    "mlp_activation": nn.SiLU(),
    # train setting
    "batch_size": 64,
    "num_workers": 16,
    "epochs": 1000,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "save_every": 10,
    "print_every": 10,
    "checkpoint_save_path": "./generated",
    # test setting
    "test_batch_size": 1,  # fixed don't change this
    "generated_path": "./dataset/cifar10_MLP/generated/generated_classifier.pth",
    "test_command": "CUDA_VISIBLE_DEVICE=0 python ./dataset/cifar10_MLP/test.py ./dataset/cifar10_MLP/generated/generated_classifier.pth"
}

# wandb
wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
wandb.init(project="cifar10_MLP", config=config)

# Data
print('==> Preparing data..')
train_set = config["dataset"](checkpoint_path=config["data_path"],
                              dim_per_token=config["dim_per_token"])
train_loader = DataLoader(dataset=train_set,
                          batch_size=config["batch_size"],
                          num_workers=config["num_workers"],
                          persistent_workers=True,
                          drop_last=True,
                          shuffle=True,)
def preprocess_data(datas):
    # print("Sequence length:", datas.size(1))
    max_length = config["max_length"]
    predict_length = config["predict_length"]
    sequence_length = config["sequence_length"]
    batch_size = datas.size(0)
    # random cutting
    random_cutting = random.randint(0, datas.size(1)-1)
    assert predict_length+max_length < sequence_length \
            or max_length >= sequence_length
    # process input
    if max_length >= sequence_length or random_cutting < max_length:
        # input length have not reach max_length
        inputs = datas[:, :random_cutting, :]
        inputs = torch.cat([model.start_padding.cpu().repeat(batch_size, 1, 1), inputs], dim=1)
    else:  # random_cutting > sequence_length-predict_length or input length reach max_length
        inputs = datas[:, random_cutting-max_length:random_cutting, :]
    # process output
    targets = datas[:, random_cutting:random_cutting+predict_length, :]
    # to return
    inputs, targets = inputs.to(config["device"], torch.float32), targets.to(config["device"], torch.float32)
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
print('==> Building diffusion..')
diffusion = DiffusionLoss(mlp_layer_dims=config["mlp_layer_dims"],
                          condition_dim=config["condition_dim"],
                          diffusion_beta_max=config["diffusion_beta_max"],
                          diffusion_n_timesteps=config["diffusion_n_timesteps"],
                          device=config["device"],
                          mlp_activation=config["mlp_activation"], )
diffusion = diffusion.to(config["device"])

# Optimizer
print('==> Building optimizer..')
optimizer = optim.AdamW(params=[{"params": model.parameters()},
                                {"params": diffusion.parameters()}],
                        lr=config["learning_rate"],
                        weight_decay=config["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                 T_max=config["epochs"])


# Training
print('==> Defining training..')

total_steps = 0
train_loss = 0
this_steps = 0
def train_one_epoch():
    global total_steps, train_loss, this_steps
    # print(f"Epoch: {epoch}")
    model.train()
    diffusion.train()
    for batch_idx, datas in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = preprocess_data(datas)
        # train
        z = model(inputs, predict_length=targets.size(1))
        loss = diffusion(targets, z)
        loss.backward()
        optimizer.step()
        # to logging losses and print and save
        train_loss += loss.item()
        wandb.log({"train_loss": loss.item()})
        this_steps += 1
        total_steps += 1
        if this_steps % config["print_every"] == 0:
            print('Loss: %.6f' % (train_loss/this_steps))
            this_steps = 0
            train_loss = 0
        if total_steps % config["save_every"] == 0:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = {"model": model.state_dict(), "diffusion": diffusion.state_dict(),
                     "optimizer": optimizer.state_dict()}
            torch.save(state, os.path.join(config["checkpoint_save_path"], "state.pth"))
            generate(save_path=config["generated_path"], need_test=True)


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    diffusion.eval()
    with torch.no_grad():
        x = model.start_padding.repeat(config["test_batch_size"], 1, 1)
        while True:
            predict_length = min(model.predict_length, config["sequence_length"] + 1 - x.size(1))
            output = model(x[:, -config["max_length"]:, :], predict_length=predict_length)
            output = diffusion.sample_ddim(x=torch.randn_like(output), z=output,
                                           sample_timesteps=100, eta=0.05)
            x = torch.cat([x, output], dim=1)
            assert x.size(1) <= config["sequence_length"] + 1
            if x.size(1) == config["sequence_length"] + 1:
                break
        prediction = x[:, 1:, :]
    train_set.save_params(prediction.cpu().to(torch.float16), save_path=save_path)
    if need_test:
        os.system(config["test_command"])
        print("\n")
    return prediction


if __name__ == '__main__':
    for epoch in range(0, config["epochs"]):
        epoch += 1
        train_one_epoch()
        scheduler.step()

    # deal problems by dataloder
    del train_loader
    exit(0)

