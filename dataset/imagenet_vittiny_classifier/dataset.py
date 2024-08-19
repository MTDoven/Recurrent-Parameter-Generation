from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os


class OneClassImageNet(Dataset):
    def __init__(self, image_root, index_file, optim_class, train):
        with open(index_file, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        self.dataset = list({os.path.join(image_root, string.split(" ")[0]): 
                             (1 if int(string.split(" ")[1]) == optim_class else 0) 
                             for i, string in enumerate(lines)}.items())
        self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]) if train else transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)



