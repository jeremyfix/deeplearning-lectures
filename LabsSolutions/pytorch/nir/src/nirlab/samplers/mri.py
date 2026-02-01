# coding: utf-8

# Standard imports
import logging
import sys
from xml.parsers.expat import model

# External imports
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
import nirlab.utils as utils

def oversample_mri(model, logdir, epoch, train_loader, oversampling_factor=1.5, batch_size=32):
    
    _, (subsampled_data, subsampled_mask, fullsampled_data) = next(iter(train_loader))

    device = next(model.parameters()).device

    subsampled_data = subsampled_data.squeeze(0).to(device)  # (kx, ky, sc, t)
    subsampled_mask = subsampled_mask.squeeze(0).to(device)  # (kx, ky)
    # fullsampled_data is unused here, but kept if you want GT comparisons

    Nrows, Ncols, Ncoils, Nframes = subsampled_data.shape

    coords = utils.build_coordinate_Nd(Nrows*oversampling_factor, Ncols*oversampling_factor, Nframes*oversampling_factor, device=device) # (Npoints, 3)
    coords = coords.to(device)

    coord_ds = torch.utils.data.TensorDataset(coords)
    coord_dl = torch.utils.data.DataLoader(coord_ds, batch_size=batch_size, shuffle=False)

    prev_training = model.training
    model.eval()

    with torch.no_grad():
        pre_intensity_list = []
        csm_list = []

        for (coord_batch,) in tqdm.tqdm(coord_dl):
            coord_batch = coord_batch.to(device).unsqueeze(0)  # (1, B, 3)
            pre_i, csm_i = model(coord_batch)  # (1, B, 1/2) and (1, B, ncoils*2)
            pre_intensity_list.append(pre_i.squeeze(0))
            csm_list.append(csm_i.squeeze(0))

        pre_intensity = torch.cat(pre_intensity_list, dim=0) # pre_intensity: (Npoints, ), csm: (Npoints, ncoils)
        csm = torch.cat(csm_list, dim=0)

        pre_intensity = pre_intensity.view(Nrows*oversampling_factor, Ncols*oversampling_factor, Nframes*oversampling_factor)  # complex
        csm = csm.view(Nrows*oversampling_factor, Ncols*oversampling_factor, Nframes*oversampling_factor, Ncoils)  # complex

        # Coil images
        recon_img = pre_intensity.unsqueeze(-1) * csm  # (kx, ky, t, sc)

        # Combine coils using sensitivities
        fused = (recon_img * torch.conj(csm)).sum(dim=-1)  # (kx, ky, t)
        img = fused.abs()
        img = img / (img.max() + 1e-12)
        img = img.cpu()
        
    if prev_training:
        model.train()

    # Plot the combined coils for every frame
    # This allows to make a video
    for frame_idx in range(img.shape[2]):
        h = plt.figure()
        plt.imshow(img[:, :, frame_idx], cmap="gray")
        plt.axis("off")
        plt.savefig(
            str(logdir / f"epoch_{epoch}_frame_{frame_idx}.png"),
            bbox_inches="tight",
        )
        plt.close(h)     
    return img

def sample_mri(model, logdir, epoch, train_loader):
    
    coords, (subsampled_data, subsampled_mask, fullsampled_data) = next(iter(train_loader))

    device = next(model.parameters()).device
    coords = coords.to(device)
    
    subsampled_data = subsampled_data.squeeze(0).to(device)  # (kx, ky, sc, t)
    subsampled_mask = subsampled_mask.squeeze(0).to(device)  # (kx, ky)
    # fullsampled_data is unused here, but kept if you want GT comparisons

    Nrows, Ncols, Ncoils, Nframes = subsampled_data.shape

    prev_training = model.training
    model.eval()

    with torch.no_grad():
        pre_intensity, csm = model(coords)  # pre_intensity: (Npoints, ), csm: (Npoints, ncoils)
        pre_intensity = pre_intensity.squeeze(0).view(Nrows, Ncols, Nframes)  # complex
        csm = csm.squeeze(0).view(Nrows, Ncols, Nframes, Ncoils)  # complex

        # Coil images
        recon_img = pre_intensity.unsqueeze(-1) * csm  # (kx, ky, t, sc)

        # Forward to k-space
        kspace_pred = utils.FFT(recon_img) # (kx, ky, t, sc) 

        # Data consistency: replace observed k-space
        kspace_pred = kspace_pred.permute(0, 1, 3, 2)  # (kx, ky, sc, t)
        mask = subsampled_mask.unsqueeze(-1).unsqueeze(-1)  # (kx, ky, 1, 1)
        mask = torch.zeros_like(mask).to(device)

        kspace_dc = kspace_pred * (1.0 - mask) + subsampled_data * mask
        kspace_dc = kspace_dc.permute(0, 1, 3, 2)  # (kx, ky, t, sc)

        # Back to image
        img_dc = utils.IFFT(kspace_dc)  # (kx, ky, t, sc)

        # Combine coils using sensitivities
        fused = (img_dc * torch.conj(csm)).sum(dim=-1)  # (kx, ky, t)
        img = fused.abs()
        img = img / (img.max() + 1e-12)
        img = img.cpu()
        
    if prev_training:
        model.train()

    # Plot the combined coils for every frame
    # This allows to make a video
    for frame_idx in range(img.shape[2]):
        h = plt.figure()
        plt.imshow(img[:, :, frame_idx], cmap="gray")
        plt.axis("off")
        plt.savefig(
            str(logdir / f"epoch_{epoch}_frame_{frame_idx}.png"),
            bbox_inches="tight",
        )
        plt.close(h)     
    return img