import torch

diction1 = torch.load("dataset/cifar10_googlenet_6m/generated/generated_classifier.pth")
diction2 = torch.load("dataset/cifar10_googlenet_6m/checkpoint/0110.pth")

def norm(x):
    if len(x.shape) == 0:
        return x
    return (x - x.mean()) / x.std()


for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    # print(key1, (norm(value1)-norm(value2)).abs().mean())
    if "running_var" in key1:
        print(value1[:8], "\n", value2[:8])
        print()

