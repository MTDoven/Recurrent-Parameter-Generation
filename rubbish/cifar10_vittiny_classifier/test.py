try:  # relative import
    from .train import *
except:  # absolute import
    from train import *
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
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

    total_target, total_predict = test(save_name=None)
    auc = roc_auc_score(total_target, total_predict)
    f1 = f1_score(total_target, (total_predict >= 0.5).astype(int))
    print("\nAUC:", auc, "\nF1:", f1)