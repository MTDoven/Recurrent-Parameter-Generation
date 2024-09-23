import random
import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")

import pandas as pd
import numpy as np
import torch
import dataset.cifar100_resnet18bn.prepare as item
loader = item.test_loader
model = item.model
test = item.test

checkpoint_path = "./dataset/cifar100_resnet18/checkpoint"
generated_path = "./dataset/cifar100_resnet18/noised"




# load paths
checkpoint_items = [os.path.join(checkpoint_path, i) for i in os.listdir(checkpoint_path)]
generated_items = [os.path.join(generated_path, i) for i in os.listdir(generated_path)]
generated_items.sort()
total_items = list(checkpoint_items) + list(generated_items)


@torch.no_grad()
def compute_wrong_indices(checkpoint):
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=False)
    model.eval()
    _, acc, all_targets, all_predicts = test(model=model)
    not_agreement = torch.logical_not(torch.eq(torch.tensor(all_targets), torch.tensor(all_predicts)))
    # if acc < 0.7618:
    #     return None, not_agreement
    return not_agreement

def compute_wrong_iou(a, b):
    a = np.logical_not(a)
    b = np.logical_not(b)
    inter = np.logical_and(a, b)
    union = np.logical_or(a, b)
    iou = np.sum(inter) / np.sum(union)
    return iou


# compute result
total_result_num = 0
print("\n==> start evaluating..")
total_result_list = []
for i, item in enumerate(total_items):
    print(f"start: {i+1}/{len(total_items)}")
    result = compute_wrong_indices(item)
    if isinstance(result, tuple) and i >= len(checkpoint_items):
        print("dropped!")
        continue
    elif isinstance(result, tuple):
        result = result[1]
    result = result.numpy()
    total_result_list.append(result)
    total_result_num += 1

# compute iou_metrix
print("Start Computing IoU...")
iou_matrix = np.zeros(shape=[total_result_num, total_result_num])
for i in range(len(total_result_list)):
    for j in range(len(total_result_list)):
        iou = compute_wrong_iou(total_result_list[i], total_result_list[j])
        iou_matrix[i, j] = iou


# save result
df = pd.DataFrame(iou_matrix)
df.to_excel("./similarity.xlsx", index=False)
print("Finished!!!")




# print ori_ori
print("num_checkpoint:", len(checkpoint_items))
print("num_generated:", total_result_num-len(checkpoint_items))
ori_num = len(checkpoint_items)
ori_ori = iou_matrix[:ori_num, :ori_num]
ori_ori = (np.sum(ori_ori) - ori_num) / (ori_num * (ori_num-1))
print("ori-ori:", ori_ori)
gen_num = total_result_num-len(checkpoint_items)
gen_gen = iou_matrix[-gen_num:, -gen_num:]
gen_gen = (np.sum(gen_gen) - gen_num) / (gen_num * (gen_num-1))
print("gen-gen:", gen_gen)
ori_gen = iou_matrix[-gen_num:, :ori_num]
ori_gen = np.mean(ori_gen)
print("ori-gen:", ori_gen)
nearest = iou_matrix[-gen_num:, :ori_num]
nearest = np.amax(nearest, axis=-1).mean()
# argmax = np.argmax(iou_matrix[-gen_num:, :ori_num], axis=-1)
# for i in argmax:
#     item = checkpoint_items[i]
#     item = torch.load(item, map_location="cpu")
#     new_diction = {}
#     for k, v in item.items():
#         if "4" in k and "bn" in k:
#             v += torch.randn_like(v) * random.randint(100, 1000) / 10000.
#         new_diction[k] = v
#     torch.save(new_diction, f"/home/wangkai/arpgen/AR-Param-Generation/dataset/cifar100_resnet18/noised/noised_{i}.pth")
print("nearest:", nearest)
