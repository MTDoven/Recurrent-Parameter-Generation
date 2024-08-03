import torch
from torch import nn
import os

# import torch
#
# # 假设我们有一个包含均匀分布数据的张量
# data = torch.rand(1000)  # 生成一个包含1000个随机数的张量
#
# # 计算中位数
# median_index = int(data.numel() / 2)  # 中位数的索引
# median_value = data.view(-1).kthvalue(median_index + 1)[0]  # kthvalue是基于0索引的，所以+1
#
# print(median_value)


diction = torch.load("./checkpoint/0110_acc0.9323_seed20_resnet18.pth")
save_file = "./generated/generated_classifier.pth"

new_diction = {}
for key, value in diction.items():
    # conv / down_sample / bn / fc
    # 0 / 0.0001 / 0.001 / 0.01 / 0.1
    if ("bn" in key) and (not ("num_batches_tracked" in key)) and ("running_var" in key):
    #if ("conv" in key) or ("downsample.0" in key):
    #if ("fc" in key):
        #print(key, value.shape)
        pre_mean = value.mean() * 0.95
        value = torch.log(value / pre_mean + 0.05)
        mean, std = value.mean(), value.std()
        #value = (value - mean) / std

        import numpy as np
        import matplotlib.pyplot as plt
        plt.clf()
        plt.hist(value.numpy().flatten(), bins=50)
        plt.savefig(f"./{key}_hist.png")
        #
        value += torch.randn_like(value) * 0.5
        #value = value * std + mean
        value = torch.clip(torch.exp(value) - 0.05, min=0.001) * pre_mean

        print(mean, std)
    new_diction[key] = value

torch.save(new_diction, save_file)
os.system(f"python test.py {save_file}")