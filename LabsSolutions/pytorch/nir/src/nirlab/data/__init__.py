# coding: utf-8

# Standard imports
from enum import nonmember
import random

# External imports
import torch
import tqdm
import numpy as np

# Local imports
from .image import ImageDataset

def build_dataset(cls, params):
    return eval(f"{cls}(**params)")

class NormalizedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, normalizing_stats):
        super().__init__()
        self.dataset = dataset
        (self.mu_x, self.std_x), (self.mu_y, self.std_y) = normalizing_stats

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return (x - self.mu_x)/self.std_x, (y - self.mu_y)/self.std_y

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, mu_x={self.mu_x}, std_x={self.std_x}, mu_y={self.mu_y}, std_y={self.std_y})"

    def __len__(self):
        return len(self.dataset)

def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_x = None
    mean2_x = None
    mean_y = None
    mean2_y = None
    num_minibatches = len(loader) # approximation
    for X, y in tqdm.tqdm(loader):
        if mean_x is None:
            mean_x = torch.zeros(X.shape[1])
            mean2_x = torch.zeros(X.shape[1])
        if mean_y is None:
            mean_y = torch.zeros(y.shape[1])
            mean2_y = torch.zeros(y.shape[1])

        mean_x += X.mean(axis=0) / num_minibatches 
        mean2_x += (X**2).mean(axis=0) / num_minibatches 
        mean_y += y.mean(axis=0) / num_minibatches 
        mean2_y += (y**2).mean(axis=0) / num_minibatches 

    std_x = torch.sqrt(mean2_x - mean_x**2)
    std_y = torch.sqrt(mean2_y - mean_y**2)

    return (mean_x.numpy(), std_x.numpy()), (mean_y.numpy(), std_y.numpy())

def get_dataloaders(config: dict, use_cuda):
    batch_size = config["batch_size"]
    normalize = config["normalize"]
    valid_ratio = config["valid_ratio"]
    num_workers = config["num_workers"]

    # We load the dataset used for training
    train_valid_dataset = build_dataset(
        config["class"],
        config["params"]
    )
    dim_input = train_valid_dataset.dim_input
    dim_output = train_valid_dataset.dim_output

    # Split the data into a train and valid fold
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    # nb_valid = int(valid_ratio * len(train_valid_dataset))
    indices = list(range(len(train_valid_dataset)))
    random.shuffle(indices)
    train_indices = indices[:nb_train]
    valid_indices = indices[nb_train:]

    train_dataset = torch.utils.data.Subset(train_valid_dataset, 
                                            train_indices)
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, 
                                            valid_indices)

    if normalize:
        normalizing_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        normalizing_stats = compute_mean_std(normalizing_loader)

        # When we are requested to normalize the data, 
        # we wrap the dataset with this normalization
        train_dataset = NormalizedDataset(train_dataset,
                                          normalizing_stats)
        valid_dataset = NormalizedDataset(valid_dataset,
                                          normalizing_stats)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, dim_input, dim_output, normalizing_stats
