import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")

# torch
import torch
# father
from workspace.main import vittiny_8192 as item
Dataset = item.Dataset
train_loader = item.train_loader
optimizer = item.optimizer
train_set = item.train_set
config = item.config
model = item.model
config["tag"] = config.get("tag") if config.get("tag") is not None else os.path.basename(item.__file__)[:-3]


test_config = {
    "device": "cuda",
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
}
config.update(test_config)




# Model
print('==> Building model..')
model.load_state_dict(torch.load(config["checkpoint"]))
model = model.to(config["device"])


# test
print('==> Defining training..')
def memory_test():
    print("==> start training..")
    model.train()
    for batch_idx, (param, permutation_state) in enumerate(train_loader):
        optimizer.zero_grad()
        # noinspection PyArgumentList
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            loss = model(output_shape=param.shape,
                         x_0=param.to(model.device),
                         permutation_state=permutation_state.to(model.device))
        loss.backward()
        optimizer.step()
        if batch_idx >= 10:
            break
    os.system("nvidia-smi")
    input(f"This program running on: {os.environ['CUDA_VISIBLE_DEVICES']}")


if __name__ == "__main__":
    memory_test()