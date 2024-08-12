import torch

diction1 = torch.load("/dataset/cifar10_vittiny_classifier/generated/generated_model_class0.pth")
diction2 = torch.load("/dataset/cifar10_vittiny_classifier/checkpoint/0016_class0_acc0.9953_vittiny.pth")

def norm(x):
    if len(x.shape) == 0:
        return x
    return (x - x.mean()) / x.std()



for (key1, value1), (key2, value2) in zip(diction1.items(), diction2.items()):
    if key1 != "head.0.bias":
        print(key1, norm(value1.flatten()[:5]), "\n",
              key1, norm(value2.flatten()[:5]))
