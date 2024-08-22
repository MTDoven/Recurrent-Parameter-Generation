
class Cifar10_ViTTiny_Classifier(ConditionalDataset):
    dataset_config = "./dataset/Cifar10/config.json"
    data_path = "./dataset/cifar10_vittiny_classifier/checkpoint"
    generated_path = "./dataset/cifar10_vittiny_classifier/generated/generated_model_class{}.pth"
    test_command = "python ./dataset/cifar10_vittiny_classifier/test.py " + \
                   "./dataset/cifar10_vittiny_classifier/generated/generated_model_class{}.pth"

    def __init__(self, checkpoint_path=None, dim_per_token=8192, **kwargs):
        super().__init__(checkpoint_path=checkpoint_path, dim_per_token=dim_per_token, **kwargs)
        # load dataset_config
        with open(self.dataset_config, "r") as f:
            dataset_config = json.load(f)
        # train dataset
        self.dataset = CIFAR10(root=dataset_config["dataset_root"], train=True, transform=None)
        self.indices = [[] for _ in range(10)]
        for index, (_, label) in enumerate(self.dataset):
            self.indices[label].append(index)
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])
        # test dataset
        self.test_dataset = CIFAR10(root=dataset_config["dataset_root"], train=False, transform=None)
        self.test_indices = [[] for _ in range(10)]
        for index, (_, label) in enumerate(self.test_dataset):
            self.test_indices[label].append(index)
        self.test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def _extract_condition(self, index: int):
        optim_class = int(super()._extract_condition(index)[1][5:])
        img_index = random.choice(self.indices[optim_class])
        img = self.transform(self.dataset[img_index][0])
        return img

    def get_image_by_class_index(self, class_index):
        img_index = random.choice(self.test_indices[class_index])
        img = self.test_transform(self.test_dataset[img_index][0])
        return img