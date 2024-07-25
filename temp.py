import torch

diction1 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/cifar10_resnet18_11m/checkpoint-92-94/0100_acc0.9263_seed18_resnet18.pth")
diction2 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/cifar10_resnet18_11m/generated/generated_seedNone.pth")

def norm(x):
    if len(x.shape) == 0:
        return x
    return (x - x.mean()) / x.std()


for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    # print(key1, (norm(value1)-norm(value2)).abs().mean())
    if "running_var" in key1 or True:
        print(value1.flatten()[:8], "\n", value2.flatten()[:8])
        print()

    #break
