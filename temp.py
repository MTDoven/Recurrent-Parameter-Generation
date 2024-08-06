import torch

diction1 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/imagenet_vittiny_6m/generated/generated_evaluate_model_001.pth")
diction2 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/imagenet_vittiny_6m/checkpoint/0000_acc0.6822_seed20_vittiny.pth")

def norm(x):
    if len(x.shape) == 0:
        return x
    return (x - x.mean()) / x.std()


all = []
for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    all.append(norm(value1.flatten()))

x = torch.cat(all)
print(x.abs().mean())