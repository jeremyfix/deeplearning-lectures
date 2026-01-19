# coding: utf-8

# Standard imports
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
        self.mu, self.std = normalizing_stats

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, (y - self.mu)/self.std

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, mu={self.mu}, std={self.std})"

    def __len__(self):
        return len(self.dataset)

def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_pix = 0
    mean2_pix = 0
    num_minibatches = len(loader) # approximation
    for imgs, _ in tqdm.tqdm(loader):
        mean_pix += imgs.mean() / num_minibatches
        mean2_pix += (imgs**2).mean() / num_minibatches

    std_pix = np.sqrt(mean2_pix - mean_pix**2)

    return mean_pix.item(), std_pix.item()

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
