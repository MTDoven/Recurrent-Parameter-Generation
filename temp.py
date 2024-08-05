import torch

diction1 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/imagenet_convnext_4m/checkpoint/0002_acc0.8017_seed20_convnext.pth")
diction2 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/imagenet_convnext_4m/generated/generated_model.pth")

def norm(x):
    if len(x.shape) == 0:
        return x
    return (x - x.mean()) / x.std()


for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    # print(key1, (norm(value1)-norm(value2)).abs().mean())
    if "running_var" in key1 or True:
        print(key2)
        print(value1.flatten()[:8], "\n", value2.flatten()[:8])
        print()
    #break
