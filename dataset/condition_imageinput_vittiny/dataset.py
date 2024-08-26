import re
import sys
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


class BinaryClassifierDataset(Dataset):
    def __init__(self, root, train, optimize_class):
        optimize_class = [optimize_class,] if isinstance(optimize_class, int) else optimize_class
        self.optimize_class = optimize_class
        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy("cifar10")),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )

    def __getitem__(self, index):
        img, origin_target = self.dataset[index]
        target = 1 if origin_target in self.optimize_class else 0
        return img, target

    def __len__(self):
        return self.dataset.__len__()




def get_optimize_class():
    try:  # get string
        string = sys.argv[1]
    except IndexError:
        RuntimeError("sys.argv[1] not found")
    class_int_string = str(re.search(r'class(\d+)', string).group(1)).zfill(4)
    one_hot_string = bin(int(class_int_string))[2:].zfill(10)
    optimize_class = [index for index, i in enumerate(one_hot_string) if i == "1"]
    return list(optimize_class), class_int_string
