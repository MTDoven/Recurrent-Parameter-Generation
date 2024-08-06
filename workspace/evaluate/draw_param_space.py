import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns



pretrained_path = "/home/wangkai/arpgen/AR-Param-Generation/dataset/imagenet_vittiny_6m/checkpoint"
generated_path = "/home/wangkai/arpgen/AR-Param-Generation/dataset/imagenet_vittiny_6m/generated"
save_path = "/home/wangkai/arpgen/AR-Param-Generation/workspace/evaluate"
tsne_extra_config = {
    "max_iter": 1000,
    "perplexity": 30.0,
    "early_exaggeration": 12.0,
    # "metric": "cosine"
}




def load_checkpoint(checkpoint_path):
    checkpoint = []
    for item in os.listdir(checkpoint_path):
        item = os.path.join(checkpoint_path, item)
        diction = torch.load(item, map_location='cpu')
        param = []
        for key, value in diction.items():
            if "num_batches_tracked" not in key:
                param.append(value.flatten())
        param = torch.cat(param, dim=0).to(torch.float32)
        checkpoint.append(param)
    checkpoint = torch.stack(checkpoint)
    return checkpoint.numpy()

pretrained = load_checkpoint(pretrained_path)
generated = load_checkpoint(generated_path)


all_data = np.vstack((pretrained, generated))
labels = ['pretrained'] * len(pretrained) + ['generated'] * len(generated)
reduced_data = PCA(n_components=len(all_data)).fit_transform(all_data)
reduced_data = TSNE(n_components=2, **tsne_extra_config).fit_transform(reduced_data)
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels)
plt.title('Param Space Visualize')
plt.savefig(os.path.join(save_path, 'draw_param_space.png'), dpi=300, bbox_inches='tight')
