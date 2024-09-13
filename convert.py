import torch

model = torch.load("convnextlarge_old.pt", map_location="cpu")["model"]

new_diction = {}
for k, v in model.items():
    if k == "model.to_condition.weight":
        k = "to_condition.weight"
    elif k == "model.to_condition.bias":
        k = "to_condition.bias"
    elif k in ["criteria.net.to_condition.weight",
               "criteria.net.to_condition.bias",
               "criteria.diffusion_trainer.model.to_condition.weight",
               "criteria.diffusion_trainer.model.to_condition.bias",
               "criteria.diffusion_sampler.model.to_condition.weight",
               "criteria.diffusion_sampler.model.to_condition.bias"]:
        continue
    new_diction[k] = v
new_diction["to_permutation_state.weight"] = torch.zeros(size=[1, 8192])

torch.save(new_diction, "main_convnextlarge_16384.pt")