import sys
src = sys.argv[1]
dst = sys.argv[2]

import os
assert os.path.exists(src) and os.path.isfile(src)
assert os.path.exists(dst) and os.path.isdir(dst)

import torch




diction = torch.load(src)
save_name = os.path.join(dst, "adapter_model.bin")
torch.save(diction, save_name)
