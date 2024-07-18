import torch

diction1 = torch.load("dataset/cifar10_mlp_1m/generated/generated_classifier.pth")
diction2 = torch.load("dataset/cifar10_mlp_1m/checkpoint/0110.pth")



for key, value in diction1.items():
    print(value.flatten()[100000:100008] * 10)
    break


for key, value in diction2.items():
    print(value.flatten()[100000:100008] * 10)
    break
