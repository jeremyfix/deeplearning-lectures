# coding: utf-8

# Standard imports
import random

# External imports
import torch
import tqdm
import numpy as np

# Local imports
from .image import ImageDataset, BilinearImageDataset
from .scene import SceneDataset
from .miccai2023 import MICCAI2023

def build_dataset(cls, params):
    if cls in ["ImageDataset", "BilinearImageDataset"]:
        valid_ratio = params.pop("valid_ratio", 0.2)
        train_valid_dataset = eval(f"{cls}(**params)")        

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
        return train_dataset, valid_dataset
    else:
        train_dataset = eval(f"{cls}(**params, train=True)")
        valid_dataset = eval(f"{cls}(**params, train=False)")

        return train_dataset, valid_dataset


def get_dataloaders(config: dict, use_cuda):
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    # We load the dataset used for training
    train_dataset, valid_dataset = build_dataset(
        config["class"],
        config["params"]
    )

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

    return train_loader, valid_loader
