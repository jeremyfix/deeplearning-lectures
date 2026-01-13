# coding: utf-8

# Standard imports
import logging
import random
import sys

# External imports
import matplotlib.pyplot as plt

# Local imports
from .miccai2023 import MICCAI2023, CINEView, AccFactor, combine_coils

def plot_sample(rootdir):
    view = CINEView.SAX
    acc = AccFactor.ACC10
    dataset = MICCAI2023(rootdir=rootdir, view=view, acc_factor=acc)

    # Access one element
    idx = random.randint(0, len(dataset)-1)
    subsampled_data, subsampled_mask, fullsampled_data = dataset[idx]

    logging.info(f"I found {len(dataset)} samples in the dataset")
    logging.info(f"Took the sample {idx}")

    # The mask is (kx, ky) boolean array with values in {0, 1} 
    # indicating if the values have been observed for frequency (fi, fj))
    logging.info(f"The boolean mask is of shape {subsampled_mask.shape}")
    
    (kx, ky, sc, sz, t) = subsampled_data.shape
    # (kx, ky, sc, sz, t) = fullsampled_data.shape
       
    # Subsampled_data and fullsampled_data are (kx, ky, sc, sz, t)
    (kx, ky) = subsampled_data.shape[:2]
    n_coils = subsampled_data.shape[2]
    n_slices = subsampled_data.shape[3]
    n_frames = subsampled_data.shape[4]

    ti= 0
    combined_subimage = combine_coils(subsampled_data[:, :, :, :, ti])
    combined_image = combine_coils(fullsampled_data[:, :, :, :, ti])

    logging.info(f"There are {n_coils} coils, with {kx}x{ky} frequencies, {n_slices} slices and {n_frames} time steps")

    for slice_idx in range(n_slices):
        fig, axes = plt.subplots(nrows=1, ncols=3)

        axes[0].imshow(subsampled_mask, cmap="gray")
        axes[0].set_title("Mask in the Fourier space")
        axes[0].axis("off")

        axes[1].imshow(combined_subimage[:, :, slice_idx], cmap="gray")
        axes[1].set_title(f"Combined sub-image at slice {slice_idx}") 
        axes[1].axis("off")

        axes[2].imshow(combined_image[:, :, slice_idx], cmap="gray")
        axes[2].set_title(f"Combined image at slice {slice_idx}") 
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(f"sample_{slice_idx}.png", bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # @SOL 
    root_dir = "/opt/datasets/MICCAI/ChallengeData"
    # SOL@
    # @TEMPL
    # root_dir = "/mounts/datasets/datasets/MICCAIChallenge2023"
    # TEMPL@
    plot_sample(root_dir)
