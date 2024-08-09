import os
import numpy as np
import torch
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from train import model, test_loader, config

checkpoint_path = "./checkpoint"
generated_path = "./generated"


# load paths
checkpoint_items = [os.path.join(checkpoint_path, i) for i in os.listdir(checkpoint_path)]
generated_items = [os.path.join(generated_path, i) for i in os.listdir(generated_path)]
total_items = list(checkpoint_items) + list(generated_items)



@torch.no_grad()
def compute_wrong_indices(checkpoint):
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()
    wrong_result = []
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
        inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        result = predicted.eq(targets)
        wrong_result.extend(result.tolist())
    return np.array(wrong_result)

def compute_wrong_iou(a, b):
    a = np.logical_not(a)
    b = np.logical_not(b)
    inter = np.logical_and(a, b)
    union = np.logical_or(a, b)
    iou = np.sum(inter) / np.sum(union)
    return iou



# compute result
print("\n==> start evaluating..")
total_result_list = []
for i, item in enumerate(total_items):
    print(f"start: {i+1}/{len(total_items)}")
    result = compute_wrong_indices(item)
    total_result_list.append(result)

# compute iou_metrix
iou_matrix = np.zeros(shape=[len(total_result_list), len(total_result_list)])
for i in range(len(total_result_list)):
    for j in range(len(total_result_list)):
        iou = compute_wrong_iou(total_result_list[i], total_result_list[j])
        iou_matrix[i, j] = iou

# save and draw
np.savetxt("iou_matrix.txt", iou_matrix, delimiter=', ', fmt='%.5f')
heat_map = sns.heatmap(iou_matrix)
plt.xlabel(f"checkpoint:{len(checkpoint_items)} / generated:{len(generated_items)}")
plt.savefig("./iou_matrix.png", dpi=300, bbox_inches='tight')
