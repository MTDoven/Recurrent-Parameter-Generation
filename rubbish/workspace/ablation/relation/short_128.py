import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")
USE_WANDB = True

# other
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
# model
from einops import rearrange
import bitsandbytes as bnb
from model import TransformerDiffusion as Model
from model.diffusion import DDPMSampler, DDIMSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import AutocastKwargs
from accelerate import Accelerator
# dataset
from dataset import ImageNet_ViTTiny as Dataset
from torch.utils.data import DataLoader



config = {
    # dataset setting
    "dataset": Dataset,
    "dim_per_token": 8192,
    "sequence_length": 808,
    # train setting
    "batch_size": 4,
    "num_workers": 8,
    "total_steps": 50000,
    "learning_rate": 0.000003,
    "weight_decay": 0.0,
    "save_every": 50000//25,
    "print_every": 50,
    "autocast": lambda i: 5000 < i < 45000,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # to log
    "model_config": {
        # transformer config
        "d_condition": 1,
        "d_model": 8192,
        "nhead": 16,
        "dim_feedforward": 8192,
        "dim_head": 512,
        "num_layers": 3,
        # diffusion config
        "diffusion_batch": 512,
        "layer_channels": [1, 32, 64, 128, 64, 32, 1],
        "model_dim": 8192,
        "condition_dim": 8192,
        "kernel_size": 7,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        "forward_once": True,
    },
    "tag": "ablation_relation_short_128",
}

mask_metrix = torch.ones(size=(config["sequence_length"], config["sequence_length"])).to(torch.long)
mask_metrix = torch.triu(1 - torch.triu(mask_metrix, diagonal=128), diagonal=-128).to(torch.bool)




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
model = Model(sequence_length=config["sequence_length"])
# replace attention layer
class MaskedAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False) if inner_dim != dim else nn.Identity()
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("mask_metrix", mask_metrix[None][None])
    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        attn = q @ k.transpose(-1, -2)
        attn = torch.where(self.mask_metrix, attn, -10000.0)
        out = self.softmax(attn) @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
for i in range(len(model.model.transformer_forward.layers)):
    model.model.transformer_forward.layers[i][0] = MaskedAttention(
        dim=config["model_config"]["d_model"],
        heads=config["model_config"]["nhead"],
        dim_head=config["model_config"]["dim_head"],
    )

# Optimizer
print('==> Building optimizer..')
optimizer = bnb.optim.AdamW8bit(params=model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
scheduler = CosineAnnealingLR(optimizer=optimizer,
                              T_max=config["total_steps"])

# accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)


# wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key="b8a4b0c7373c8bba8f3d13a2298cd95bf3165260")
    wandb.init(project="AR-Param-Generation", name=config['tag'], config=config,)




# Training
print('==> Defining training..')
def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> start training..")
    model.train()
    for batch_idx, param in enumerate(train_loader):
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
            torch.save(state, os.path.join(config["checkpoint_save_path"], config["tag"]+".pth"))
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
        os.system(config["test_command"])
        print("\n")
    model.train()
    return prediction


if __name__ == '__main__':
    train()
    del train_loader  # deal problems by dataloader
    print("Finished Training!")
    exit(0)