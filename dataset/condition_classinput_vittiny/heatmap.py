import os
import torch

checkpoint_path = "./checkpoint_279"
layer = "head.0.weight"
tag = layer + "_" + checkpoint_path[2:]


def extract_param(checkpoint, layer):
    diction = torch.load(checkpoint, map_location="cpu")
    layer = diction[layer].flatten()
    return layer

param_list = []
for checkpoint in os.listdir(checkpoint_path):
    checkpoint = os.path.join(checkpoint_path, checkpoint)
    param = extract_param(checkpoint, layer)
    param_list.append(param)
params = torch.stack(param_list)




import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
params = params.clip(max=2.5).numpy()[:40, :120]
sns.heatmap(params, cmap='GnBu')
plt.savefig(f'heatmap_{tag}.png')
