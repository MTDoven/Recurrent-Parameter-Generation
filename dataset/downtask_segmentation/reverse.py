import numpy as np
import torch
import sys

file = sys.argv[1]

try:
    model = torch.load(file, map_location='cpu')['state_dict']
    torch.save(model, file)
    print('this file has been reversed.')
except KeyError:
    print('this file need not to reverse.')
