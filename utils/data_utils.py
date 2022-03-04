import logging

import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # if args.dataset == "cifar10":
    #     trainset = datasets.CIFAR10(root="./data",
    #                                 train=True,
    #                                 download=True,
    #                                 transform=transform_train)
    #     testset = datasets.CIFAR10(root="./data",
    #                                train=False,
    #                                download=True,
    #                                transform=transform_test) if args.local_rank in [-1, 0] else None
    #
    # else:
    #     trainset = datasets.CIFAR100(root="./data",
    #                                  train=True,
    #                                  download=True,
    #                                  transform=transform_train)
    #     testset = datasets.CIFAR100(root="./data",
    #                                 train=False,
    #                                 download=True,
    #                                 transform=transform_test) if args.local_rank in [-1, 0] else None

    #Train_Set
    cifar10_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    cifar100_dataset = datasets.CIFAR100(root="./data", train=True, transform=transform_train, download=True)
    sub_masks = []
    cifar10_targets = np.array(cifar10_dataset.targets)
    sub_size = 500
    cifar10_class_count = 10
    for i in range(cifar10_class_count):
        mask = np.random.choice(np.arange(len(cifar10_targets))[cifar10_targets == i], size=sub_size, replace=False)
        sub_masks.append(mask)
    sub_mask = np.sort(np.stack(sub_masks).reshape(-1))
    cifar10_dataset.targets = cifar10_targets[sub_mask]
    cifar10_dataset.data = cifar10_dataset.data[sub_mask]
    cifar100_dataset.data = np.concatenate([cifar10_dataset.data, cifar100_dataset.data])
    cifar100_dataset.targets = np.array(cifar100_dataset.targets) + cifar10_class_count
    cifar100_dataset.targets = np.concatenate([cifar10_dataset.targets, cifar100_dataset.targets])
    trainset = cifar100_dataset

    #Test Set
    cifar10_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)
    cifar100_dataset = datasets.CIFAR100(root="./data", train=False, transform=transform_test, download=True)
    sub_masks = []
    cifar10_targets = np.array(cifar10_dataset.targets)
    sub_size = 100
    cifar10_class_count = 10
    for i in range(cifar10_class_count):
        mask = np.random.choice(np.arange(len(cifar10_targets))[cifar10_targets == i], size=sub_size, replace=False)
        sub_masks.append(mask)
    sub_mask = np.sort(np.stack(sub_masks).reshape(-1))
    cifar10_dataset.targets = cifar10_targets[sub_mask]
    cifar10_dataset.data = cifar10_dataset.data[sub_mask]
    cifar100_dataset.data = np.concatenate([cifar10_dataset.data, cifar100_dataset.data])
    cifar100_dataset.targets = np.array(cifar100_dataset.targets) + cifar10_class_count
    cifar100_dataset.targets = np.concatenate([cifar10_dataset.targets, cifar100_dataset.targets])
    testset = cifar100_dataset

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
