import os
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import math


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
    def __init__(self, checkpoint_path, dim_per_token, **kwargs):
        super().__init__(checkpoint_path, **kwargs)
        self.kwargs["dim_per_token"] = dim_per_token
        self.preprocess(diction=torch.load(self.checkpoint_list[0]),
                        first_step=True, dim_per_token=dim_per_token)

    def preprocess(self, diction: dict, **kwargs) -> torch.Tensor:
        dim_per_token = kwargs['dim_per_token']
        param_list = []
        for key, value in diction.items():
            if kwargs.get("first_step"):
                self.structure[key] = value.shape
            value = pad_to_length(value, dim_per_token)
            value = value.flatten()
            value = torch.chunk(value, chunks=len(value) // dim_per_token, dim=0)
            assert len(value[0]) == dim_per_token, \
                f"This param need padding: {key} with shape of {diction[key].shape}."
            param_list.extend(list(value))
        param = torch.cat(param_list, dim=0)
        param = param.view(-1, dim_per_token)
        return param

    def postprocess(self, params: torch.Tensor, **kwargs) -> dict:
        dim_per_token = kwargs['dim_per_token']
        diction = {}
        params = params.flatten()
        for key, shape in self.structure.items():
            num_elements = math.prod(shape)
            this_param = params[:num_elements].view(*shape)
            diction[key] = this_param
            if num_elements % dim_per_token == 0:
                params = params[num_elements:]
            else:  # drop padding
                num_elements = (num_elements // dim_per_token + 1) * dim_per_token
                params = params[num_elements:]
        return diction


if __name__ == "__main__":
    dataset = Cifar10_MLP(checkpoint_path="./cifar10_MLP/checkpoint", dim_per_token=4096)
    x = dataset[0]
    print(x.shape)
    dataset.save_params(x, "./test.path")




