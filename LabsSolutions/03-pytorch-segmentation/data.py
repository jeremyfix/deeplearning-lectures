#!/usr/bin/env python3
# coding : utf-8

# Standard imports
import pathlib

# External imports
import albumentations as A
from albumentations.pytorch import ToTensorV2


class StanfordDataset:
    def __init__(self, rootdir: pathlib.Path):
        pass


def get_dataloaders(
    rootdir: pathlib.Path,
    cuda: bool,
    batch_size: int,
    n_workers: int,
    small_experiment: bool,
):
    pass
    return None, None


def test_dataloaders():
    rootdir = pathlib.Path("/opt/Datasets/stanford")
    cuda = False
    batch_size = 32
    n_workers = 4
    small_experiment = True

    train_loader, valid_loader = get_dataloaders(
        rootdir, cuda, batch_size, n_workers, small_experiment
    )


if __name__ == "__main__":

    test_dataloaders()
