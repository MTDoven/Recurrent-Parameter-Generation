from train import *


if __name__ == '__main__':
    config["test_model_path"] = "/home/mtdoven/Project/AR-Param-Generation/AR-Param-Generation/dataset/cifar10_MLP/checkpoint/300.pth"

    model.load_state_dict(torch.load(config["test_model_path"]))
    test(save_name=None)