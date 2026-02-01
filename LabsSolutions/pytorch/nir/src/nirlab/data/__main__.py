# coding: utf-8

# Standard imports
import logging
import random
import sys

# External imports
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Local imports
from .miccai2023 import MICCAI2023, CINEView, AccFactor, plot_sample
from . import get_dataloaders
from .image import ImageDataset, BilinearImageDataset


def test_image_dataset(rootdir):
    logging.info("="*80)
    scale_factor = 4

    logging.info("Testing image dataset")
    dataset = ImageDataset(rootdir)

    height = dataset.height * scale_factor
    width = dataset.width * scale_factor
    sampled_image = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            sampled_image[i, j] = dataset.sample([j/width, i/height])
    sampled_image *= 255
    sampled_image = sampled_image.astype(np.uint8)
    PIL_image = Image.fromarray(sampled_image)
    PIL_image.save("sampled_image.png")
    logging.info("Saved sampled image as sampled_image.png")

    # Let us now use the Bilinear sampler
    logging.info("="*80)
    logging.info("Testing image dataset")
    dataset = BilinearImageDataset(rootdir)
    
    height = dataset.height * scale_factor
    width = dataset.width * scale_factor
    sampled_image = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            sampled_image[i, j] = dataset.sample([j/width, i/height])
    sampled_image *= 255
    sampled_image = sampled_image.astype(np.uint8)
    PIL_image = Image.fromarray(sampled_image)
    PIL_image.save("sampled_bilinear_image.png")
    logging.info("Saved sampled image as sampled_bilinear_image.png")


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
    
    kx, ky, sc, t = subsampled_data.shape 
    # kx, ky, sc, t = fullsampled_data.shape

    logging.info(f"(kx, ky) = {kx, ky}")
    logging.info(f"Number of coils: {sc}")
    logging.info(f"Number of time frames: {t}")

    plot_sample(subsampled_data, subsampled_mask, fullsampled_data)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    # Image dataset
    rootdir = "./toyimages"
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


