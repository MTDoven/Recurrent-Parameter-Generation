import os
import sys
if __name__ == "__main__":
    from train import *
else:  # relative import
    from .train import *
from torchvision.datasets import CIFAR10
from torchvision import transforms



try:
    test_item = sys.argv[1]
except IndexError:
    assert __name__ == "__main__"
    test_item = "./generated"
test_items = []
if os.path.isdir(test_item):
    for item in os.listdir(test_item):
        item = os.path.join(test_item, item)
        test_items.append(item)
elif os.path.isfile(test_item):
    test_items.append(test_item)




original_dataset = CIFAR10(
    root=config["dataset_root"],
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)
original_targets = [original_dataset[i][1] for i in range(len(original_dataset))]
original_targets = torch.tensor(original_targets, dtype=torch.long)


for item in test_items:
    state = torch.load(item, map_location="cpu")
    model.load_state_dict({key: value.to(torch.float32).to(device) for key, value in state.items()})
    loss, acc, all_targets, all_predicts = test(model=model)
    all_targets, all_predicts = torch.tensor(all_targets), torch.tensor(all_predicts)

    for class_idx in range(10):
        class_mask = torch.where(original_targets == class_idx, 1, 0)
        total_number = torch.sum(class_mask)
        correct = torch.where(all_targets == all_predicts, 1, 0)
        class_correct = class_mask * correct
        correct_number = torch.sum(class_correct)
        class_acc = correct_number.item() / total_number.item()
        print(f"class{class_idx}:", class_acc)
