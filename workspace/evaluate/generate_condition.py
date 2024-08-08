import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")

# torch
import torch
# father
from workspace import vittiny_condition_8192 as item
Dataset = item.Dataset
train_set = item.train_set
config = item.config
model = item.model



generate_config = {
    "device": "cuda",
    "num_generated": 10,  # DO NOT CHANGE THIS LINE
    "checkpoint": f"./checkpoint/{item.__name__.split('.')[-1]}.pth",
    "generated_path": os.path.join(Dataset.generated_path.rsplit("/", 1)[0], "generated_model_class{}.pth"),
    "test_command": os.path.join(Dataset.test_command.rsplit("/", 1)[0], "generated_model_class{}.pth"),
    "need_test": True,
}
config.update(generate_config)



# Model
print('==> Building model..')
model.load_state_dict(torch.load(config["checkpoint"]))
model = model.to(config["device"])


# generate
print('==> Defining generate..')
def generate(save_path=config["generated_path"], need_test=True, class_index=None):
    print("\n==> Generating..")
    model.eval()
    condition = train_set.get_image_by_class_index(class_index=class_index)[None]
    with torch.no_grad():
        prediction = model(sample=True, condition=condition)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    train_set.save_params(prediction, save_path=save_path.format(class_index))
    if need_test:
        os.system(config["test_command"].format(class_index)+f" {class_index}")
        print("\n")
    model.train()
    return prediction


if __name__ == "__main__":
    for i in range(config["num_generated"]):
        index = str(i+1).zfill(3)
        generate(
            save_path=config["generated_path"],
            need_test=config["need_test"],
            class_index=i,
        )
        print("\n\n")