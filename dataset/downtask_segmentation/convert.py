import numpy as np
import torch
import sys
import os

file = sys.argv[1]
model = torch.load(file, map_location="cpu")
if 'meta' in model.keys():
    print("this file need not to convert.")
    exit(0)
else:  # this is a raw checkpoint
    meta_file = os.path.join(os.path.dirname(__file__), "Segmentation/example.pth")
    meta_data = torch.load(meta_file, map_location="cpu")['meta']
    model = {'meta': meta_data, "state_dict": model}
    torch.save(model, file)
    print("converted to test-able file.")
