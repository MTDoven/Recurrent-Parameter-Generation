try:
    from .train import *
except ImportError:
    from train import *
import sys


if __name__ == '__main__':
    config["test_model_path"] = sys.argv[1]

    model.load_state_dict(torch.load(config["test_model_path"]))
    test(save_name=None)