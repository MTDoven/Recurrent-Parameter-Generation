import os
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import math
import sys


class BaseDataset(Dataset, ABC):
    def __init__(self, checkpoint_path, dim_per_token, **kwargs):
        assert os.path.exists(checkpoint_path)
        self.dim_per_token = dim_per_token
        checkpoint_list = os.listdir(checkpoint_path)
        self.checkpoint_list = list([os.path.join(checkpoint_path, item) for item in checkpoint_list])
        self.length = self.real_length = len(self.checkpoint_list)
        self.structure = {}
        diction = torch.load(self.checkpoint_list[0], map_location="cpu")
        for key, value in diction.items():
            self.structure[key] = value.shape
        self.kwargs = kwargs

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.real_length
        diction = torch.load(self.checkpoint_list[index], map_location="cpu")
        return self.preprocess(diction)

    def save_params(self, params, save_path):
        diction = self.postprocess(params.cpu())
        torch.save(diction, save_path)

    def set_infinite_dataset(self, max_num=None):
        if max_num is None:
            max_num = self.length * 1000000
        self.length = max_num
        return self

    @abstractmethod
    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        pass


def pad_to_length(x, common_factor):
    if len(x.flatten()) % common_factor == 0:
        return x.flatten()
    # print(f"padding {x.shape} according to {common_factor}")
    full_length = (len(x.flatten()) // common_factor + 1) * common_factor
    padding_length = full_length - len(x.flatten())
    padding = torch.zeros([padding_length, ], dtype=x.dtype, device=x.device)
    x = torch.cat((x.flatten(), padding), dim=0)
    return x


class Cifar10_MLP(BaseDataset):
    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        param_list = []
        for key, value in diction.items():
            # print(key, value.shape)
            value = value.flatten()
            param_list.append(value)
        param = torch.cat(param_list, dim=0)
        param = pad_to_length(param, self.dim_per_token)
        param = param.view(-1, self.dim_per_token)
        # print("Sequence length:", param.size(0))
        return param

    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        diction = {}
        params = params.flatten()
        for key, shape in self.structure.items():
            num_elements = math.prod(shape)
            this_param = params[:num_elements].view(*shape)
            diction[key] = this_param
            params = params[num_elements:]
        return diction


class ImageDebugDataset(Dataset):
    def __init__(self, checkpoint_path, dim_per_token, **kwargs):
        self.length = self.real_length = 1
        self.dim_per_token = dim_per_token
        self.kwargs = kwargs
        from PIL import Image
        from torchvision.transforms import ToTensor, ToPILImage
        img = ToTensor()(Image.open(os.path.join(checkpoint_path, "image.jpg")))
        self.img = img - img.mean()
        self.shape = self.img.shape
        self.mean = img.mean()
        self.to_pil_image = ToPILImage()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        param = pad_to_length(self.img, self.dim_per_token)
        param = param.view(-1, self.dim_per_token)
        return param

    def save_params(self, params, save_path):
        params = params.flatten()
        num_elements = math.prod(self.shape)
        img = params[:num_elements].view(*self.shape)
        img = img + self.mean
        img = self.to_pil_image(img)
        img.save(save_path)

    def set_infinite_dataset(self, max_num=None):
        if max_num is None:
            max_num = self.length * 10000000
        self.length = max_num
        return self




if __name__ == "__main__":
    dataset = ImageDebugDataset(checkpoint_path="image_debug_2m/checkpoint", dim_per_token=256, predict_length=1)
    x = dataset[0]
    print(x.shape)
    dataset.save_params(x, "./test.pth")
    dataset.set_infinite_dataset()
    print(dataset[100000])
    print(len(dataset))
    os.remove("./test.pth")

