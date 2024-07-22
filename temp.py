import torch

diction1 = torch.load("dataset/cifar10_mobilenet_2m/generated/generated_classifier.pth")
diction2 = torch.load("dataset/cifar10_mobilenet_2m/checkpoint/0110.pth")



for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    print((value1-value2).norm())

