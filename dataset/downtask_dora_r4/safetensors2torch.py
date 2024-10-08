import sys
src = sys.argv[1]
dst = sys.argv[2]

import os
assert os.path.exists(src) and os.path.isdir(src)
assert os.path.exists(dst) and os.path.isdir(dst)

from safetensors.torch import safe_open
from tqdm import tqdm
import torch




def load(folder):
    diction = {}
    with safe_open(os.path.join(folder, "model.safetensors"), framework='pt') as f:
        for k in f.keys():
            v = f.get_tensor(k)
            diction[k] = v
    return diction


for folder in tqdm(os.listdir(src)):
    folder = os.path.join(src, folder)
    diction = load(folder)
    save_name = os.path.join(dst, f"{folder.split('-')[-1]}.pth")
    torch.save(diction, save_name)
