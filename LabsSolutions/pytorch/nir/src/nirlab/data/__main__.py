# coding: utf-8

# Standard imports
import logging
import random
import sys

# External imports
import matplotlib.pyplot as plt

# Local imports
from .miccai2023 import MICCAI2023, CINEView, AccFactor, plot_sample
from . import get_dataloaders
from .image import BilinearImageDataset


def test_image_dataset(rootdir):
    logging.info("="*80)
    logging.info("Testing image dataset")
    dataset = BilinearImageDataset(rootdir)
    logging.info(f"Dataset input dimension: {dataset.dim_input}")
    logging.info(f"Dataset output dimension: {dataset.dim_output}")
    logging.info(f"Dataset size: {len(dataset)}")

    logging.info("Trying to index the dataset...")
    idx = random.randint(0, len(dataset)-1)
    pos, value = dataset[idx]
    logging.info(f"Dataset {idx} : at position {pos}, value is {value}")


def test_image_dataloaders():
    logging.info("="*80)
    logging.info("Testing image dataloaders")
    config = {
        "class": "ImageDataset",
        "params": {
            "root_dir": "."
        },
        "batch_size": 32,
        "normalize": False,
        "valid_ratio": 0.2,
        "num_workers": 4
    }

    use_cuda = False

    train_loader, valid_loader = get_dataloaders(config, use_cuda)

    logging.info(f"Number of training batches: {len(train_loader)}")
    logging.info(f"Number of validation batches: {len(valid_loader)}")

def test_miccai_dataset(rootdir):
    logging.info("="*80)
    logging.info("Testing MICCAI2023 dataset")

    view = CINEView.SAX
    acc = AccFactor.ACC10
    dataset = MICCAI2023(rootdir=rootdir, view=view, acc_factor=acc, slice_idx=0, train=True, valid_frames=[4, 6])

    # Access one element
    idx = 0
    coords, (subsampled_data, subsampled_mask, fullsampled_data) = dataset[idx]

    logging.info(f"I found {len(dataset)} samples in the dataset")
    logging.info(f"Took the sample {idx}")

    # The mask is (kx, ky) boolean array with values in {0, 1} 
    # indicating if the values have been observed for frequency (fi, fj))
    logging.info(f"The boolean mask is of shape {subsampled_mask.shape}")
    
    (kx, ky, sc, t, _) = subsampled_data.shape # _ is 2
    # (kx, ky, sc, t, _) = fullsampled_data.shape # _ is 2

    logging.info(f"(kx, ky) = {kx, ky}")
    logging.info(f"Number of coils: {sc}")
    logging.info(f"Number of time frames: {t}")

    plot_sample(subsampled_data, subsampled_mask, fullsampled_data)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    # Image dataset
    rootdir = "./images"
    test_image_dataset(rootdir)
    test_image_dataloaders()

    # MICCAI2023 dataset
    # @SOL 
    root_dir = "/opt/datasets/MICCAI/ChallengeData"
    # SOL@
    # @TEMPL
    # root_dir = "/mounts/datasets/datasets/MICCAIChallenge2023"
    # TEMPL@
    test_miccai_dataset(root_dir)


