import torch
import torchvision.transforms as transforms

import os
import os.path
import numpy as np
import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_transforms(dataset_name):
    # Common transformations
    if 'cifar10' in dataset_name:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    elif 'cifar100' in dataset_name:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    elif dataset_name == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
    elif 'imagenet' in dataset_name:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return transform_train, transform_test

class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def prepare_dataset(dataset_name, batch_size, load_train=True, num_workers=4, data_dir='./data'):    
    """Prepare the dataset based on the provided name and return data loaders."""
    transform_train, transform_test = get_transforms(dataset_name)
    transform = transform_train if load_train else transform_test
    if dataset_name == 'cifar10':
        if load_train:
            dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        if load_train:
            dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name == 'svhn':
        if load_train:
            dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
        else:
            dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)
    elif dataset_name == 'imagenet':
        if load_train:
            datadir = os.path.join(data_dir, 'train')
        else:
            datadir = os.path.join(data_dir, 'val')
        dataset = datasets.ImageFolder(datadir, transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
