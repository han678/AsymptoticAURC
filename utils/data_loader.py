from __future__ import print_function

import os

import models as models
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def prepare_dataset(dataset_name, train_batch, test_batch, num_workers):
    """Prepare the dataset based on the provided name and return data loaders."""
    # Common transformations
    if dataset_name == 'cifar10':
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
    elif dataset_name == 'cifar100':
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
    elif dataset_name == 'imagenet':
        root = '../datasets/ILSVRC2012/'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'val')
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

    # Dataset-specific handling
    if dataset_name == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'cifar100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'svhn':
        trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    elif dataset_name == 'imagenet':
        trainset = datasets.ImageFolder(traindir, transform_train)
        testset = datasets.ImageFolder(testdir, transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=num_workers)

    return trainloader, testloader