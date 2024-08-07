import torch

diction1 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/cifar10_vittiny_condition/generated/generated_model_class3.pth")
diction2 = torch.load("/home/wangkai/arpgen/AR-Param-Generation/dataset/cifar10_vittiny_condition/checkpoint/0016_class3_acc0.9873_vittiny.pth")

def norm(x):
    if len(x.shape) == 0:
        return x
    return (x - x.mean()) / x.std()



for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    print(key1, norm(value1.flatten()[:5]), norm(value2.flatten()[:5]))
