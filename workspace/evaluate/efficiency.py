import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")

# torch
import time
import types
import torch
import copy
from torch import nn
from model.diffusion import DDIMSampler, DDPMSampler
# father
from workspace.main import convnextatto_8192 as item
Dataset = item.Dataset
train_set = item.train_set
config = item.config
model = item.model
config["tag"] = config.get("tag") if config.get("tag") is not None else os.path.basename(item.__file__)[:-3]


generate_config = {
    "device": "cuda",
    "num_generated": 1,
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
    "generated_path": os.path.join(Dataset.generated_path.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "test_command": os.path.join(Dataset.test_command.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "need_test": True,
}
config.update(generate_config)




# Model
print('==> Building model..')
model.load_state_dict(torch.load(config["checkpoint"]))
model.criteria.diffusion_sampler = DDIMSampler(
    model=model.criteria.diffusion_sampler.model,
    beta=config["model_config"]["beta"],
    T=config["model_config"]["T"],
)
model.condi_embedder = copy.deepcopy(model.criteria.diffusion_sampler.model.condi_embedder)
@torch.no_grad()
def new_sample(self, x=None, condition=None):
    z = self.model([1, self.sequence_length, self.config["d_model"]], condition)
    z = self.condi_embedder(z)
    if x is None:
        x = torch.randn((1, self.sequence_length, self.config["model_dim"]), device=z.device)
    x = self.criteria.sample(x, z)  # steps=200, eta=0.1)
    return x
model.sample = types.MethodType(new_sample, model)
model.criteria.diffusion_sampler.model.condi_embedder = nn.Identity()
model = model.to(config["device"])




# generate
print('==> Defining generate..')
def generate(save_path=config["generated_path"], test_command=config["test_command"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.cuda.amp.autocast(True, torch.bfloat16):
        with torch.no_grad():
            start_time = time.time()
            prediction = model(sample=True)
            end_time = time.time()
            generated_norm = torch.nanmean(prediction.abs())
    print("used time:", end_time - start_time)
    print("memory usage:", torch.cuda.max_memory_allocated() / (1024 ** 3))
    print("Generated_norm:", generated_norm.item())
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(test_command)
        print("\n")


if __name__ == "__main__":
    for i in range(config["num_generated"]):
        index = str(i+1).zfill(3)
        print("Save to", config["generated_path"].format(config["tag"], index))

        generate(
            save_path=config["generated_path"].format(config["tag"], index),
            test_command=config["test_command"].format(config["tag"], index),
            need_test=config["need_test"],
        )
