import sys, os
sys.path.append("/home/wangkai/arpgen/AR-Param-Generation")
os.chdir("/home/wangkai/arpgen/AR-Param-Generation")

# torch
import torch
# father
from workspace.main.sequence import vittiny_4096 as item
Dataset = item.Dataset
train_set = item.train_set
config = item.config
model = item.model



generate_config = {
    "device": "cuda",
    "num_generated": 3,
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
    "generated_path": os.path.join(Dataset.generated_path.rsplit("/", 1)[0], "generated_{}_acc{}_{}.pth"),
    "test_command": os.path.join(Dataset.test_command.rsplit("/", 1)[0], "generated_{}_acc{}_{}.pth"),
    "need_test": True,
}
config.update(generate_config)



# Model
print('==> Building model..')
model.load_state_dict(torch.load(config["checkpoint"]))
model = model.to(config["device"])


# generate
print('==> Defining generate..')
def generate(save_path=config["generated_path"], test_command=config["test_command"], need_test=True, condition=None):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        prediction = model(sample=True, condition=condition)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(test_command)
        print("\n")


if __name__ == "__main__":
    for i in range(config["num_generated"]):
        index = str(i+1).zfill(3)
        for performance in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
            print(config['generated_path'].format(config['tag'], performance, index))
            generate(
                save_path=config["generated_path"].format(config["tag"], performance, index),
                test_command=config["test_command"].format(config["tag"], performance, index),
                need_test=config["need_test"],
                condition=torch.tensor(performance, dtype=torch.float32)
            )
        print("\n\n")