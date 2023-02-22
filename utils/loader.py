import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def dataset_loader(train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(size=[32,32], padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2470, 0.2435, 0.2616]),
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(0.3, 1/0.3), value='random')
        ])
        dataset = datasets.CIFAR10(root='/data/dataset', train=True, download=True, transform=transform)

        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size - train_size)
        print(f'Dataset size: Train({train_size}), Val({val_size})')

        return torch.utils.data.random_split(dataset, [train_size, val_size])

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2470, 0.2435, 0.2616])
        ])
        dataset = datasets.CIFAR10(root='/data/dataset', train=False, download=False, transform=transform)
        test_size = len(dataset)
        print(f'Dataset size: Test({test_size})')

        return dataset
