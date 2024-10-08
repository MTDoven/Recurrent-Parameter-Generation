import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# torch
import torch
# father
from workspace.main import convnextlarge_16384 as item
Dataset = item.Dataset
train_set = item.train_set
config = item.config
model = item.model
config["tag"] = config.get("tag") if config.get("tag") is not None else os.path.basename(item.__file__)[:-3]


generate_config = {
    "device": "cuda",
    "num_generated": 10,
    "checkpoint": f"./checkpoint/{config['tag']}.pth",
    "generated_path": os.path.join(Dataset.generated_path.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "test_command": os.path.join(Dataset.test_command.rsplit("/", 1)[0], "generated_{}_{}.pth"),
    "need_test": True,
}
config.update(generate_config)



# Model
print('==> Building model..')
model.load_state_dict(torch.load(config["checkpoint"]))
model = model.to(config["device"])


# generate
print('==> Defining generate..')
def generate(save_path=config["generated_path"], test_command=config["test_command"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.cuda.amp.autocast(True, torch.bfloat16):
        with torch.no_grad():
            prediction = model(sample=True)
            generated_norm = torch.nanmean(prediction.abs())
    print("Generated_norm:", generated_norm.item())
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(test_command)
        print("\n")


if __name__ == "__main__":
    for i in range(config["num_generated"]):
        index = str(i+1).zfill(3)
        print("Save to", config["generated_path"].format(config["tag"], index))
        generate(
            save_path=config["generated_path"].format(config["tag"], index),
            test_command=config["test_command"].format(config["tag"], index),
            need_test=config["need_test"],
        )