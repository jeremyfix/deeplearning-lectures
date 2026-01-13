# coding: utf-8

# Standard imports
import logging
import random
import sys

# Local imports
from .miccai2023 import MICCAI2023, CINEView, AccFactor

def plot_sample(rootdir):
    view = CINEView.SAX
    acc = AccFactor.ACC4
    dataset = MICCAI2023(rootdir=rootdir, view=view, acc_factor=acc)

    # Access one element
    idx = random.randint(0, len(dataset))
    subsampled_data, subsampled_mask, fullsampled_data = dataset[idx]

    logging.info(f"I found {len(dataset)} samples in the dataset")
    logging.info(f"Took the sample {idx}")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # @SOL 
    root_dir = "/opt/datasets/MICCAI/"
    # SOL@
    # @TEMPL
    # root_dir = "/mounts/datasets/datasets/MICCAIChallenge2023"
    # TEMPL@
    plot_sample(root_dir)
