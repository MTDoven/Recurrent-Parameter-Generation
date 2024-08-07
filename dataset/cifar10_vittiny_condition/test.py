try:  # relative import
    from .train import *
except:  # absolute import
    from train import *
import sys


if __name__ == '__main__':
    config["test_model_path"] = sys.argv[1]
    config["batch_size"] = 50

    state = torch.load(config["test_model_path"], map_location="cpu")
    diction = {}
    for key, value in state.items():
        diction[key] = value.to(torch.float32)
    model.load_state_dict(diction)
    model = model.to(config["device"])

    test(save_name=None)