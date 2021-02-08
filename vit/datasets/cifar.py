import torch

from torchvision import transforms, datasets
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)


def cifar(args, distributed=False):

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (28, 28), scale=(0.05, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=transform_test,
            )
        )

    else:
        trainset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR100(
                root="./data",
                train=False,
                download=True,
                transform=transform_test,
            )
        )


    train_sampler = (
        RandomSampler(trainset)
    )
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=args.val_batch_size,
            num_workers=4,
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, '_', test_loader, '_'
