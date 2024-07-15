import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn import functional as F
import wandb
from torch.utils.data import DataLoader
from model import BiARModule, RNNModule, DiffusionLoss
from dataset import Cifar10_MLP
import random
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


config = {
    # device setting
    "device": "cuda:6",
    # dataset setting
    "dataset": Cifar10_MLP,
    "data_path": "./dataset/cifar10_MLP_middle/checkpoint",
    "dim_per_token": 2048,
    "sequence_length": 5056,
    # model config
    "num_layers": 12,
    "input_length": 64,
    "predict_length": 64,
    "token_mixer_ks": 65,
    "feed_forward_ks": 5,
    "activation": nn.ELU(),
    # diffusion loss config
    "mlp_layer_dims": [2048, 2048, 2048, 2048],
    "condition_dim": 2048,
    "diffusion_beta_max": 0.02,
    "diffusion_n_timesteps": 1000,
    # train setting
    "batch_size": 1,
    "num_workers": 8,
    "total_steps": 200000,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "save_every": 500,
    "print_every": 20,
    "warmup_steps": 250,
    "checkpoint_save_path": "./generated",
    # test setting
    "test_batch_size": 1,  # fixed don't change this
    "generated_path": "./dataset/cifar10_MLP_middle/generated/generated_classifier.pth",
    "test_command": "CUDA_VISIBLE_DEVICE=5 python ./dataset/cifar10_MLP_middle/test.py ./dataset/cifar10_MLP_middle/generated/generated_classifier.pth"
}

# wandb
wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
wandb.init(project="cifar10_MLP", config=config)

# Data
print('==> Preparing data..')
train_set = config["dataset"](checkpoint_path=config["data_path"],
                              dim_per_token=config["dim_per_token"],
                              predict_length=config["predict_length"],
                              fix_one_sample=True,)
print("Dataset length:", train_set.length)
train_loader = DataLoader(dataset=train_set,
                          batch_size=config["batch_size"],
                          num_workers=config["num_workers"],
                          persistent_workers=True,
                          drop_last=True,
                          shuffle=True,)
def preprocess_data(datas):
    # print("Sequence length:", datas.size(1)); exit(0)
    max_length = config["input_length"]
    predict_length = config["predict_length"]
    config["sequence_length"] = datas.size(1)
    assert max_length % predict_length == 0
    assert datas.size(1) % predict_length == 0
    random_cutting = random.randint(0, datas.size(1) // predict_length - 1) * predict_length
    inputs = datas[:, max(random_cutting-max_length, 0):random_cutting, :]
    targets = datas[:, random_cutting:random_cutting+predict_length, :]
    inputs, targets = inputs.to(config["device"], torch.float32), targets.to(config["device"], torch.float32)
    return inputs, targets

# Model
print('==> Building model..')
# model = BiARModule(num_layers=config["num_layers"],
#                    hidden_dim=config["dim_per_token"],
#                    input_length=config["input_length"],
#                    predict_length=config["predict_length"],
#                    token_mixer_ks=config["token_mixer_ks"],
#                    feed_forward_ks=config["feed_forward_ks"],
#                    activation=config["activation"],)
model = RNNModule(hidden_dim=config["dim_per_token"],
                  book_size=config["sequence_length"]//config["predict_length"],
                  out_conv_ks=config["feed_forward_ks"])
model = model.to(config["device"])

# Loss
print('==> Building diffusion..')
diffusion = DiffusionLoss(mlp_layer_dims=config["mlp_layer_dims"],
                          condition_dim=config["condition_dim"],
                          diffusion_beta_max=config["diffusion_beta_max"],
                          diffusion_n_timesteps=config["diffusion_n_timesteps"],
                          mlp_activation=config["activation"],
                          device=config["device"],)
diffusion = diffusion.to(config["device"])

# Optimizer
print('==> Building optimizer..')
optimizer = optim.RMSprop(params=[{"params": model.parameters()},
                                  {"params": diffusion.parameters()}],
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
    # print(f"Epoch: {epoch}")
    model.train()
    diffusion.train()
    for batch_idx, datas in enumerate(train_loader):
        optimizer.zero_grad()
        # inputs, targets = preprocess_data(datas)
        # train
        # z = model(inputs)
        assert datas.size(1) % config["predict_length"] == 0
        num_slices = datas.size(1) // config["predict_length"]
        predicts = []
        predict, state = model(last_state=None, batch_size=datas.size(0))
        predicts.append(predict)
        for _ in range(num_slices-1):
            predict, state = model(state, batch_size=datas.size(0))
            predicts.append(predict)
        prediction = torch.cat(predicts, dim=1).flatten(start_dim=1)
        targets = datas.flatten(start_dim=1).to(config["device"], torch.float32)
        # loss = F.mse_loss(z, targets)  # FIXME: use diffusion loss
        loss = F.mse_loss(prediction, targets)
        # loss = diffusion(targets, z)
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
            state = {"model": model.state_dict(), "diffusion": diffusion.state_dict(),
                     "optimizer": optimizer.state_dict()}
            torch.save(state, os.path.join(config["checkpoint_save_path"], "state.pth"))
            generate(save_path=config["generated_path"], need_test=True)
        if total_steps >= config["total_steps"]:
            break


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    diffusion.eval()
    with torch.no_grad():
        # x = torch.zeros(config["test_batch_size"], 0, config["dim_per_token"], device=config["device"])
        # while True:
        #     z = model(x[:, -config["input_length"]:, :])
        #     output = z  # FIXME: use diffusion loss
        #     # output = diffusion.sample(x=torch.randn_like(z), z=z,
        #     #                           sample_timesteps=100, eta=0.05)
        #     x = torch.cat([x, output], dim=1)
        #     assert x.size(1) < config["sequence_length"] + config["predict_length"]
        #     if x.size(1) >= config["sequence_length"]:
        #         break
        num_slices = config["sequence_length"] // config["predict_length"]
        predicts = []
        predict, state = model(last_state=None, batch_size=1)
        predicts.append(predict)
        for _ in range(num_slices-1):
            predict, state = model(state, batch_size=1)
            predicts.append(predict)
        prediction = torch.cat(predicts, dim=1).flatten(start_dim=1)
        prediction = prediction.view(prediction.size(0), -1, config["dim_per_token"])
        # prediction = x[:, :config["sequence_length"], :]
    print("Generated_norm:", prediction.abs().mean())
    #print("Generated_norm:", z.abs().mean())
    train_set.save_params(prediction.cpu().to(torch.float16), save_path=save_path)
    if need_test:
        os.system(config["test_command"])
        print("\n")
    model.train()
    diffusion.train()
    return prediction


if __name__ == '__main__':
    # model = torch.compile(model, mode="default")
    train()

    # deal problems by dataloder
    del train_loader
    print("Finished Training!")
    exit(0)

