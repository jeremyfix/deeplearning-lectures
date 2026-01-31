# coding: utf-8

# Standard imports
import logging
import sys

# External imports
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tqdm
import matplotlib.pyplot as plt

# Local imports
import nirlab.utils as utils

def sample_mri(model, logdir, epoch):
    
    # Nrows, Ncols, Nt
    Nrows, Ncols, Nframes = 100, 100, 20
    coords = utils.build_coordinate_Nd(Nrows, Ncols, Nframes).unsqueeze(0) # (1, Npoints, 3)

    prev_training = model.training
    model.eval()

    with torch.no_grad():
        # pre_intensity, csm = model(coords)  # pre_intensity: (Npoints, ), csm: (Npoints, ncoils)
        # Ncoils = csm.shape[-1]

        # pre_intensity = pre_intensity.view(Nrows, Ncols, Nframes, 1)  # (Nrows, Ncols, Nframes, 1)
        # csm = csm.view(Nrows, Ncols, Nframes, Ncoils)  # (Nrows, Ncols, Nframes, Ncoils)

        # # Apply the same pre-instensity through every coil specific sensitivity
        # recon_img = pre_intensity * csm  # (Nrows, Ncols, Nframes, Ncoils)

        # # Merge all the coils, each contribution being modulated by the CSM
        # fused_recon_img = (recon_img * torch.conj(csm)).sum(axis=-1)

        # # Plot the reconstruction with the contributions of all the coils
        # scale_factor = fused_recon_img.abs().max()

        # img = fused_recon_img.abs() / scale_factor
        # img = img.cpu()  #  (Nrows, Ncols, Nframes)    
        pre_intensity, csm = model(coords)  # pre_intensity: (Npoints, ), csm: (Npoints, ncoils)
        pre_intensity = pre_intensity.view(Nrows, Ncols, Nframes)
        pre_intensity = pre_intensity.abs()
        scale_factor = pre_intensity.max()
        img = pre_intensity / scale_factor
        img = img.cpu()  #  (Nrows, Ncols, Nframes)
        
    if prev_training:
        model.train()

    # Plot the combined coils for every frame
    # This allows to make a video
    for frame_idx in range(img.shape[2]):
        h = plt.figure()
        plt.imshow(img[:, :, frame_idx], cmap="gray")
        plt.savefig(
            str(logdir / f"epoch_{epoch}_frame_{frame_idx}.png"),
            bbox_inches="tight",
        )
        plt.close(h)     
    return img