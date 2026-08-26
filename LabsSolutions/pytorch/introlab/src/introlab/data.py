# coding: utf-8

# Standard imports
import logging
import random

# External imports
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.utils import make_grid
import PIL

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

class WrappedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        xi, yi = self.dataset[idx]
        t_xi = self.transform(xi)
        return t_xi, yi

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, transform={self.transform})"

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    root_dir = data_config["root_dir"]
    normalize = data_config["normalize"]

    logging.info("  - Dataset creation")

    # @SOL
    base_dataset = torchvision.datasets.FashionMNIST(
        root=root_dir,
        train=True,
        download=True,
        transform=None
    )
    # SOL@
    # @TEMPL
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # # TODO: Create the FashionMNIST dataset
    # #       The variable rootdir is useful
    # base_dataset = None
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@

    logging.info(f"  - I loaded {len(base_dataset)} samples")

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    num_valid = int(valid_ratio * len(base_dataset))
    train_indices = indices[num_valid:]
    valid_indices = indices[:num_valid]

    # @SOL
    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)
    # SOL@
    # @TEMPL
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # # TODO : Create the train and valid splits. The torch.utils.data.Subset
    # #        class is useful for this purpose
    # train_dataset = None
    # valid_dataset = None
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@

    preprocess_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]
    # @SOL
    if normalize:
        preprocess_transforms.append(v2.Normalize(mean=[0.2860], std=[0.2750]))
    # SOL@
    augmentation_transforms = [
        # @SOL
        # v2.RandomHorizontalFlip(),
        # v2.RandomRotation(10),
        # SOL@
        # v2.AutoAugment(),  # @SOL@
    ]

    train_transforms = v2.Compose(preprocess_transforms + augmentation_transforms)
    train_dataset = WrappedDataset(train_dataset, train_transforms)

    valid_transforms = v2.Compose(preprocess_transforms)
    valid_dataset = WrappedDataset(valid_dataset, valid_transforms)

    # Build the dataloaders
    # @TEMPL
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # # TODO: Create the train and valid dataloaders
    # # from their respective datasets
    # train_loader = None
    # valid_loader = None
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@
    # @SOL
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    # SOL@
    num_classes = len(base_dataset.classes)
    input_size = tuple(train_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes, base_dataset.classes

def display_samples(loader, nsamples, classes, filename=None):
    # Get one minibatch, hopefully containing at least nsamples samples :)
    imgs, labels = next(iter(loader))
    # Convert the class indices to labels
    labels = [classes[i] for i in labels]

    fig, axes = plt.subplots(1, nsamples, figsize=(20, 5))

    for xi, yi, axi in zip(imgs, labels, axes):
        axi.imshow(xi[0], vmin=0, vmax=1.0, cmap=cm.gray)
        axi.set_title(yi, fontsize=15)
        axi.get_xaxis().set_visible(False)
        axi.get_yaxis().set_visible(False)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    # plt.show()

def plot_samples():
    root_dir = "data"
    batch_size = 32
    use_cuda = torch.cuda.is_available()
    preprocess_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = torchvision.datasets.FashionMNIST(
        root=root_dir,
        train=True,
        download=True,
        transform=preprocess_transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    display_samples(loader, 10, dataset.classes, 'fashionMNIST_samples.png')

def test_dataloaders():
    # @TEMPL
    # pass 
    # TEMPL@
    # @SOL
    data_config = {
        "root_dir": "./data",
        "valid_ratio": 0.2,
        "batch_size": 32,
        "num_workers": 0,
    }
    use_cuda = torch.cuda.is_available()

    train_loader, valid_loader, input_size, num_classes, classes = get_dataloaders(
        data_config, use_cuda
    )

    X, y = next(iter(train_loader))
    grid = make_grid(X, nrow=8)
    show(grid)
    plt.tight_layout()
    plt.savefig("fashionMNIST_grid.png", bbox_inches='tight')
    # plt.show()
    # SOL@

# @SOL
def plot_augmented_samples():
    root_dir = "./data"
    use_cuda = torch.cuda.is_available()
    preprocess_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    augmentation_transforms = [
        # v2.RandomHorizontalFlip(),
        # v2.RandomRotation(10),
        # v2.RandomResizedCrop(128, scale=(0.8, 1.0)),
        # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.TrivialAugmentWide(),  # @SOL@
    ]

    transform = v2.Compose(preprocess_transforms + augmentation_transforms)

    dataset = torchvision.datasets.FashionMNIST(
        root=root_dir,
        train=True,
        download=True,
        transform=transform
    )

    fig, axs = plt.subplots(3, 3)
    idx = 1000
    _, y = dataset[idx]
    print(f"The sample is a {dataset.classes[y]}")
    for axi in axs.ravel():
        x, y = dataset[idx]
        axi.imshow(x[0].numpy(), cmap=cm.gray)
        axi.axis("off")
    plt.tight_layout()
    plt.savefig("fashionMNIST_aug_samples.png", bbox_inches="tight", pad_inches=0)
# SOL@

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_samples()
    test_dataloaders()
    plot_augmented_samples()  # @SOL@
