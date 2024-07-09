import os
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import math
import sys


def pad_to_length(x, common_factor):
    if len(x.flatten()) % common_factor == 0:
        return x.flatten()
    full_length = (len(x.flatten()) // common_factor + 1) * common_factor
    padding_length = full_length - len(x.flatten())
    padding = torch.zeros([padding_length, ], dtype=x.dtype, device=x.device)
    x = torch.cat((x.flatten(), padding), dim=0)
    return x


class BaseDataset(Dataset, ABC):
    def __init__(self, checkpoint_path, **kwargs):
        assert os.path.exists(checkpoint_path)
        checkpoint_list = os.listdir(checkpoint_path)
        self.checkpoint_list = list([os.path.join(checkpoint_path, item) for item in checkpoint_list])
        self.length = len(self.checkpoint_list)
        self.structure = {}
        self.kwargs = kwargs

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.kwargs.get("fix_one_sample"):
            index = 0
        diction = torch.load(self.checkpoint_list[index], map_location="cpu")
        return self.preprocess(diction, **self.kwargs)

    def save_params(self, params, save_path):
        diction = self.postprocess(params.cpu(), **self.kwargs)
        torch.save(diction, save_path)

    @abstractmethod
    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        pass


class Cifar10_MLP(BaseDataset):
    def __init__(self, checkpoint_path, dim_per_token, predict_length, **kwargs):
        super().__init__(checkpoint_path, **kwargs)
        self.dim_per_token = dim_per_token
        self.predict_length = predict_length
        self.norm = [0.0, 1.0]
        self.preprocess(diction=torch.load(self.checkpoint_list[0]),
                        first_step=True, dim_per_token=dim_per_token)
        if kwargs.get("fix_one_sample"):
            self.length = 640000000

    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        param_list = []
        for key, value in diction.items():
            if kwargs.get("first_step"):
                self.structure[key] = value.shape
            value = value.flatten()
            param_list.append(value)
        param = torch.cat(param_list, dim=0)
        param = pad_to_length(param, self.dim_per_token * self.predict_length)
        param = param.view(-1, self.dim_per_token)
        # print("Sequence length:", param.size(0))
        if kwargs.get("first_step") and self.kwargs.get("fix_one_sample"):
            self.norm[0], self.norm[1] = param.mean(), param.std()
        param = (param - self.norm[0]) / self.norm[1]
        return param

    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        diction = {}
        params = params.flatten()
        params = params * self.norm[1] + self.norm[0]
        for key, shape in self.structure.items():
            num_elements = math.prod(shape)
            this_param = params[:num_elements].view(*shape)
            diction[key] = this_param
            params = params[num_elements:]
        return diction


if __name__ == "__main__":
    dataset = Cifar10_MLP(checkpoint_path="cifar10_MLP_middle/checkpoint", dim_per_token=4096, predict_length=32)
    x = dataset[0]
    print(x.shape)
    dataset.save_params(x, "./test.pth")

