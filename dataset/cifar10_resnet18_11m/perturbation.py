import torch
from torch import nn
import os


diction = torch.load("./checkpoint-one/0146_acc0.9400_seed20_resnet18.pth")
save_file = "./generated/generated_classifier.pth"

new_diction = {}
for key, value in diction.items():
    # conv / down_sample / bn / fc
    # 0 / 0.0001 / 0.001 / 0.01 / 0.1
    if ("bn" in key) and (not ("num_batches_tracked" in key)) and ("running_var" in key):
    #if ("conv" in key) or ("downsample.0" in key):
    #if ("fc" in key):
        print(key)
        mean, std = value.mean(), value.std()
        value = (value - mean) / std
        value += torch.randn_like(value) * 1.0
        value = value * std + mean
        if "running_var" in key:
            value = torch.clip(value, min=1e-8)
    new_diction[key] = value

torch.save(new_diction, save_file)
os.system(f"python test.py {save_file}")