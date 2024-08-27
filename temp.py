import torch

diction1 = torch.load("/data/personal/nus-wk/arpgen/AR-Param-Generation/dataset/cifar10_vittiny/generated/generated_model.pth")
diction2 = torch.load("/data/personal/nus-wk/arpgen/AR-Param-Generation/dataset/cifar10_vittiny/generated/generated_model.pth")

def norm(x):
    if len(x.shape) == 0:
        return x
    return (x - x.mean()) / x.std()



for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    if "running_var" in key1:
        print("ok")
        print(key1)




