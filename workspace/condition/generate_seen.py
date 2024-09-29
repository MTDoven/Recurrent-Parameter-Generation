import sys, os
sys.path.append("/mnt/petrelfs/zhaowangbo.p/arpgen/AR-Param-Generation")
os.chdir("/mnt/petrelfs/zhaowangbo.p/arpgen/AR-Param-Generation")

# torch
import torch
import random
# father
from workspace.condition import generalization as item
train_set = item.train_set
test_set = item.test_set
train_set.set_infinite_dataset(max_num=train_set.real_length)
print("num_generated:", test_set.real_length)
config = item.config
model = item.model
config["tag"] = config.get("tag") if config.get("tag") is not None else os.path.basename(item.__file__)[:-3]


generate_config = {
    "device": "cuda",
    "checkpoint": "/mnt/petrelfs/zhaowangbo.p/arpgen/AR-Param-Generation/checkpoint/generalization.pth",  # f"./checkpoint/{config['tag']}.pth",
    "generated_path": os.path.join(test_set.generated_path.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "test_command": os.path.join(test_set.test_command.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "need_test": True,
    "specific_item": 37, 
}
config.update(generate_config)



# Model
print('==> Building model..')
model.load_state_dict(torch.load(config["checkpoint"]))
model = model.to(config["device"])


# generate
print('==> Defining generate..')
def generate(save_path=config["generated_path"], test_command=config["test_command"], need_test=True, index=None):
    print("\n==> Generating..")
    model.eval()
    _, condition = train_set[index]
    class_index = str(int("".join([str(int(i)) for i in condition]), 2)).zfill(4)
    with torch.no_grad():
        prediction = model(sample=True, condition=condition[None], permutation_state=False)
        generated_norm = torch.nanmean((prediction.cpu()).abs())
    print("Generated_norm:", generated_norm.item())
    train_set.save_params(prediction, save_path=save_path.format(config["tag"], f"class{class_index}"))
    if need_test:
        os.system(test_command.format(config["tag"], f"class{class_index}"))
        print("\n")
    model.train()
    return prediction


if __name__ == "__main__":
    for i in range(20):
        if config["specific_item"] is not None:
            assert isinstance(config["specific_item"], int)
            i = config["specific_item"]
            print(f"generate index {i}\n")
        print("Save to", config["generated_path"].format(config["tag"], "classXXX"))
        generate(
            save_path=config["generated_path"],
            test_command=config["test_command"],
            need_test=config["need_test"],
            index=random.randint(0, len(train_set)-1) if config["specific_item"] is None else i,
        )