import sys, os
sys.path.append("/data/personal/nus-wk/arpgen/AR-Param-Generation")
os.chdir("/data/personal/nus-wk/arpgen/AR-Param-Generation")
USE_WANDB = True

# other
import dill
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
import bitsandbytes as bnb
# model
from mamba_ssm import Mamba2 as Mamba
from model.mamba import MambaModel
from model import MambaDiffusion as Model
from model.diffusion import DDPMSampler, DDIMSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import AutocastKwargs
from accelerate import Accelerator
# dataset
from dataset import ImageNet_Convnext as Dataset
from torch.utils.data import DataLoader



config = {
    # dataset setting
    "dataset": Dataset,
    "dim_per_token": 16384,
    "sequence_length": 'auto',
    # train setting
    "resume": False,
    "batch_size": 1,
    "num_workers": 4,
    "total_steps": 100000,
    "learning_rate": 0.00001,
    "weight_decay": 1e-5,
    "save_every": 100000//25,
    "print_every": 50,
    "autocast": lambda i: 10000 < i < 90000,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_device": 0,
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # to log
    "model_config": {
        # mamba config
        "d_condition": 1,
        "pre_d_model": 8192,
        "pre_d_state": 128,
        "d_model": 16384,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 2,
        # diffusion config
        "diffusion_batch": 1024,
        "layer_channels": [1, 64, 96, 64, 1],
        "model_dim": 16384,
        "condition_dim": 16384,
        "kernel_size": 7,
        "sample_mode": DDIMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        "forward_once": True,
    },
    "tag": 'superlarge_convnext_16384',
}




# Data
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=config["dim_per_token"])
print("Dataset length:", train_set.real_length)
print("input shape:", train_set[0].shape)
if config["sequence_length"] == "auto":
    config["sequence_length"] = train_set.sequence_length
    print(f"sequence length: {config['sequence_length']}")
else:  # set fix sequence_length
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
model = Model(sequence_length=config["sequence_length"])  # model setting is in model
class VaryMambaModel(nn.Module):
    config = {}
    def __init__(self, sequence_length):
        super().__init__()
        mamba1 = Mamba(d_model=config["model_config"]["pre_d_model"],
                       d_state=config["model_config"]["pre_d_state"],
                       d_conv=config["model_config"]["d_conv"],
                       expand=config["model_config"]["expand"])
        mamba2 = Mamba(d_model=config["model_config"]["d_model"],
                       d_state=config["model_config"]["d_state"],
                       d_conv=config["model_config"]["d_conv"],
                       expand=config["model_config"]["expand"])
        mamba2.in_proj = nn.Linear(mamba1.out_proj.out_features, mamba2.in_proj.out_features, bias=False)
        self.mamba_forward = nn.Sequential(*[mamba1, mamba2])
        self.to_condition = nn.Linear(self.config["d_condition"], config["model_config"]["pre_d_model"])
        pe = self.get_sinusoid(sequence_length, config["model_config"]["pre_d_model"])[None, :, :]
        if self.config.get("trainable_pe"):
            self.pe = nn.Parameter(pe)
        else:  # fixed positional embedding
            self.register_buffer("pe", pe)
    @staticmethod
    def get_sinusoid(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    def forward(self, output_shape, condition=torch.tensor([0.])):
        condition = self.to_condition(condition.view(-1, 1, self.config["d_condition"]).to(self.pe.device))
        x = self.mamba_forward(self.pe.repeat(output_shape[0], 1, 1) + condition)
        return x
VaryMambaModel.config = config["model_config"]
model.model = VaryMambaModel(sequence_length=config["sequence_length"])
torch.cuda.empty_cache()

# Optimizer
print('==> Building optimizer..')
optimizer = bnb.optim.AdamW8bit(params=model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
scheduler = CosineAnnealingLR(optimizer=optimizer,
                              T_max=config["total_steps"])

# load checkpoint
if config["resume"] and os.path.exists("./vitbase_state.pt"):
    diction = torch.load("./vitbase_state.pt", map_location="cpu")
    model.load_state_dict(diction["model"])
    optimizer.load_state_dict(diction["optimizer"])
    scheduler.load_state_dict(diction["scheduler"])
    start_batch_idx = diction["step"] + 1
else:  # not resume
    start_batch_idx = 0

# accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(
        project="AR-Param-Generation",
        name=config["tag"],
        config=config,
        resume=config["resume"],
    )




# Training
print('==> Defining training..')
def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> start training..")
    model.train()
    for batch_idx, param in enumerate(train_loader):
        batch_idx += start_batch_idx
        optimizer.zero_grad()
        # train
        # noinspection PyArgumentList
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            loss = model(output_shape=param.shape, x_0=param)
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            scheduler.step()
        # to logging losses and print and save
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"train_loss": loss.item()})
        elif USE_WANDB:
            pass  # don't print
        else:  # not use wandb
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx % config["save_every"] == 0 and accelerator.is_main_process:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = accelerator.unwrap_model(model).state_dict()
            torch.save(state, os.path.join(config["checkpoint_save_path"],
                                           f"{__file__.split('/')[-1].split('.')[0]}.pth"))
            torch.save({
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": accelerator.unwrap_model(optimizer).state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": batch_idx
            }, "./vitbase_state.pt")
            generate(save_path=config["generated_path"], need_test=True)
        if batch_idx >= config["total_steps"]:
            break


def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    # _, condition = train_set[0]
    with torch.no_grad():
        prediction = model(sample=True)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    if USE_WANDB and accelerator.is_main_process:
        wandb.log({"generated_norm": generated_norm.item()})
    if accelerator.is_main_process:
        train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(f"CUDA_VISIBLE_DEVICES={config['test_device']} " + config["test_command"])
        print("\n")
    model.train()
    return prediction


if __name__ == '__main__':
    train()
    del train_loader  # deal problems by dataloader
    print("Finished Training!")
    exit(0)
