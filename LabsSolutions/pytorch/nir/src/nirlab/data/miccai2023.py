# coding: utf-8

# This file originates and is adapted from the torchcvnn python library
# https://github.com/torchcvnn/torchcvnn

# MIT License

# Copyright (c) 2024 Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
from enum import Enum
import pathlib
import logging
from typing import Union
import random

# External imports
import torch
from torch.utils.data import Dataset
import h5py  # Required because the data are matlab v7.3 files
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from nirlab import utils


class CINEView(Enum):
    SAX = 1
    LAX = 2


class AccFactor(Enum):
    ACC4 = 4
    ACC8 = 8
    ACC10 = 10


def load_matlab_file(filename: str, key: str) -> np.ndarray:
    """
    Load a matlab file in HDF5 format
    """
    with h5py.File(filename, "r") as f:
        logging.debug(f"Got the keys {f.keys()} from {filename}")
        data = f[key][()]

    return data


def kspace_to_image(kspace: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Convert k-space data to image data. The returned kspace is
    of the same type than the the provided image (np.ndarray or torch.Tensor).

    Arguments:
        kspace : torch.Tensor or np.ndarray
            k-space data

    Returns:
        torch.Tensor or np.ndarray
            image data
    """
    if isinstance(kspace, torch.Tensor):
        img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace)))
    else:
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
    return img


def image_to_kspace(
    img: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert image data to k-space data. The returned kspace is
    of the same type than the the provided image (np.ndarray or torch.Tensor)

    Arguments:
        img : torch.Tensor or np.ndarray
            Image data

    Returns:
        torch.Tensor or np.ndarray
            k-space data

    """
    if isinstance(img, torch.Tensor):
        kspace = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))
    else:
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    return kspace


def combine_coils_from_kspace(kspace: np.ndarray) -> np.ndarray:
    """
    Combine the coils of the k-space data using the root sum of squares

    Arguments:
        kspace : np.ndarray
            k-space data of shape (sc, ky, kx)

    Returns:
        np.ndarray
            Image data with coils combined, of shape (ky, kx), real valued, positive
    """
    if kspace.ndim != 3:
        raise ValueError(
            f"kspace should have 3 dimensions, got {kspace.ndim}. Expected dimensions (sc, ky, kx)"
        )
    images = np.fft.ifft2(np.fft.ifftshift(kspace))
    return np.fft.fftshift(np.sqrt(np.sum(np.abs(images) ** 2, axis=0)))

@torch.jit.script
def IFFT(x):
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.fftshift(x, dim=(0, 1)), dim=(0, 1)), dim=(0, 1)
    )

def combine_coils(kspace):
    """
    Combine the coils from the given k-space

    Arguments:
        kspace: Tensor of shape (nrows, ncols, ncoils)
                     or (nrows, ncols, ncoils, nslices)
                complex valued

    Returns:
        image: Tensor of shape (nrows, ncols)
                    or (nrows, ncols, nslices)
                magnitude only
    """
    if isinstance(kspace, np.ndarray):
        kspace = torch.tensor(kspace, dtype=torch.complex64)

    images = IFFT(kspace)

    # Combine the coils in the image space with the RSS
    coils_combined = (images.abs() ** 2).sum(axis=2).sqrt()

    return coils_combined

class MICCAI2023(Dataset):
    """
    Loads the MICCAI2023 challenge data for the reconstruction task Task 1

    The data are described on https://cmrxrecon.github.io/Task1-Cine-reconstruction.html

    You need to download the data before hand in order to use this class.

    For loading the data, you may want to alternatively consider the fastmri library, see https://github.com/facebookresearch/fastMRI/

    The structure of the dataset is as follows:

        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/P{id}/
                                    - cine_sax.mat
                                    - cin_lax.mat
        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/AccFactor04/P{id}/
                                    - cine_sax.mat
                                    - cine_sax_mask.mat
                                    - cin_lax.mat
                                    - cine_lax_mask.mat
        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/AccFactor08/P{id}/
                                    - cine_sax.mat
                                    - cine_sax_mask.mat
                                    - cin_lax.mat
                                    - cine_lax_mask.mat
        rootdir/ChallengeData/MultiCoil/cine/TrainingSet/AccFactor10/P{id}/
                                    - cine_sax.mat
                                    - cine_sax_mask.mat
                                    - cin_lax.mat
                                    - cine_lax_mask.mat

    The cine_sax or sine_lax files are :math:`(k_x, k_y, s_c, s_z, t)` where :

    - :math:`k_x`: matrix size in x-axis (k-space)
    - :math:`k_y``: matrix size in y-axis (k-space)
    - :math:`s_c`: coil array number (compressed to 10)
    - :math:`s_x`: matrix size in x-axis (image)
    - :math:`s_y`: matrix size in y-axis (image) , used in single-coil data
    - :math:`s_z`: slice number for short axis view, or slice group for long axis (i.e., 3ch, 2ch and 4ch views)
    - :math:`t`: time frame.

    Note the k-space dimensions (in x/y axis) are not the same depending on the patient.

    This is a recontruction dataset. The goal is to reconstruct the fully sampled k-space
    from the subsampled k-space. The acceleratation factor specifies the subsampling rate.

    There are also the Single-Coil data which is not yet considered by this implementation
    """

    def __init__(
        self,
        rootdir: str,
        view: CINEView = CINEView.SAX,
        acc_factor: AccFactor = AccFactor.ACC4,
        patient_idx: int = 0,
        train: bool = True,
        slice_idx: int = 0,
        valid_frames: list = [],
    ):
        self.rootdir = pathlib.Path(rootdir)

        if view == CINEView.SAX:
            self.input_filename = "cine_sax.mat"
            self.mask_filename = "cine_sax_mask.mat"
        elif view == CINEView.LAX:
            self.input_filename = "cine_lax.mat"
            self.mask_filename = "cine_lax_mask.mat"

        # List all the available data
        self.fullsampled_rootdir = self.rootdir / "MultiCoil" / "cine" / "TrainingSet"
        self.fullsampled_key = "kspace_full"
        self.subsampled_rootdir = (
            self.rootdir
            / "MultiCoil"
            / "cine"
            / "TrainingSet"
            / f"AccFactor{acc_factor.value:02d}"
        )
        self.subsampled_key = f"kspace_sub{acc_factor.value:02d}"
        self.mask_key = f"mask{acc_factor.value:02d}"

        self.train = train
        self.slice_idx = slice_idx
        self.valid_frames = valid_frames

        logging.info(f"Loading data from {self.subsampled_rootdir}")

        # We list all the patients in the subsampled data directory
        # and check we have the data, mask and full sampled data
        self.patients = []
        for patient in self.subsampled_rootdir.iterdir():
            if not patient.is_dir():
                continue

            if not (patient / self.input_filename).exists():
                logging.warning(f"Missing {self.input_filename} for patient {patient}")
                continue

            if not (patient / self.mask_filename).exists():
                logging.warning(f"Missing {self.mask_filename} for patient {patient}")
                continue

            fullsampled_patient = self.fullsampled_rootdir / patient.name
            if not (fullsampled_patient / self.input_filename).exists():
                logging.warning(
                    f"Missing {self.input_filename} for patient {fullsampled_patient}"
                )
                continue

            self.patients.append(patient)

        logging.debug(
            f"I found {len(self.patients)} patient(s) : {[p.name for p in self.patients]}"
        )

        # Load the requested patient
        self.load_patient(patient_idx)

    def load_patient(self, patient_idx: int):
        """
        Load all the data for a given patient index

        Arguments:
            patient_idx: Index of the patient to load
        """
        assert 0 <= patient_idx < len(self.patients), f"patient_idx should be in [0, {len(self.patients)-1}]"

        patient = self.patients[patient_idx]

        # Load the subsampled data
        logging.info(f"Loading {patient / self.input_filename}")
        subsampled_data = load_matlab_file(
            patient / self.input_filename, self.subsampled_key
        ).transpose(3, 4, 2, 1, 0)
        subsampled_data = subsampled_data["real"] + 1j * subsampled_data["imag"]
        subsampled_data = torch.tensor(subsampled_data)
        # (kx, ky, sc, sz, t) for multi-coil data
        # e.g. (246, 512, 10, 10, 12)    

        # Loading the mask    
        logging.info(f"Loading {patient / self.mask_filename}")
        subsampled_mask = load_matlab_file(
            patient / self.mask_filename, self.mask_key
        ).transpose(0, 1)
        subsampled_mask = torch.tensor(subsampled_mask)
        # (kx, ky)
        # e.g. (246, 512)   
            
        # Load the fully sampled data
        logging.info(
            f"Loading {self.fullsampled_rootdir / patient.name / self.input_filename}"
        )
        fullsampled_data = load_matlab_file(
            self.fullsampled_rootdir / patient.name / self.input_filename,
            self.fullsampled_key,
        ).transpose(3, 4, 2, 1, 0)
        fullsampled_data = fullsampled_data["real"] + 1j * fullsampled_data["imag"]
        fullsampled_data = torch.tensor(fullsampled_data)
        # kx, ky, sc, sz, t
        # e.g. (246, 512, 10, 10, 12)   
        
        # Precompute the coordinates
        nrows, ncols, ncoils, nslices, nframes = subsampled_data.shape
        logging.info(f"Number of rows {nrows}, cols {ncols}, coils {ncoils}, slices {nslices}, frames {nframes}")

        # Normalize the data for this slice
        # This step is super important for the training to work properly !!!!
        coils_combined = combine_coils(subsampled_data)
        norm_factor = coils_combined.max()
        logging.info(f"Normalizing the data with factor {norm_factor.item():.6f}")

        # Only keep the requested slices
        logging.info(f"Only keeping the slice index {self.slice_idx}")
        subsampled_data = subsampled_data[:, :, :, self.slice_idx, :] # (kx, ky, sc, t)
        fullsampled_data = fullsampled_data[:, :, :, self.slice_idx, :] # (kx, ky, sc, t)

        frames_to_keep = []
        if self.train:
            for t in range(nframes):
                if t not in self.valid_frames:
                    frames_to_keep.append(t)
        else:
            frames_to_keep = self.valid_frames

        logging.info(f"Keeping frames {frames_to_keep}")
        subsampled_data = subsampled_data[:, :, :, frames_to_keep] # (kx, ky, sc, t)
        fullsampled_data = fullsampled_data[:, :, :, frames_to_keep]

        # Normalize the data for this slice
        # This step is super important for the training to work properly !!!!
        #coils_combined = combine_coils(subsampled_data)
        #norm_factor = coils_combined.max()
        subsampled_data = subsampled_data / norm_factor.item()
        # And use the same factor for the fully sampled data
        fullsampled_data = fullsampled_data / norm_factor.item()

        # Assign these data to the instance members
        self.subsampled_data = subsampled_data
        self.subsampled_mask = subsampled_mask
        self.fullsampled_data = fullsampled_data

        # Also filter the coordinates given the frames to keep
        row_lin = torch.linspace(0, 1, nrows)
        col_lin = torch.linspace(0, 1, ncols)
        time_lin = torch.linspace(0, 1, nframes)[frames_to_keep]
        coords_mesh = torch.meshgrid(row_lin, col_lin, time_lin, indexing="ij")
        self.coords = torch.stack(coords_mesh, -1).view(-1, 3)

    def __len__(self):
        return 1 # Handles only one patient at a time

    def __getitem__(self, idx):
        """
        Returns the subsampled k-space data, the mask and the fully sampled k-space data
        """
        return self.coords, (self.subsampled_data, self.subsampled_mask, self.fullsampled_data)

def plot_sample(subsampled_data, subsampled_mask, fullsampled_data): 
    """
    Plot a sample of the MICCAI2023 dataset

    It shows the mask in k-space, the combined sub-sampled image and the combined fully sampled image
    """    
    # Subsampled_data and fullsampled_data are (kx, ky, sc, t)
    kx, ky, n_coils, n_frames = subsampled_data.shape
    n_slices = 1
    ti= 0

    combined_subimage = combine_coils(subsampled_data[:, :, :, ti])
    combined_image = combine_coils(fullsampled_data[:, :, :, ti])

    logging.info(f"There are {n_coils} coils, with {kx}x{ky} frequencies, {n_slices} slices and {n_frames} time steps")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    axes[0].imshow(subsampled_mask, cmap="gray")
    axes[0].set_title("Mask in the Fourier space", fontsize=10, pad=10)
    axes[0].axis("off")

    axes[1].imshow(combined_subimage, cmap="gray")
    axes[1].set_title("Combined sub-image", fontsize=10, pad=10) 
    axes[1].axis("off")

    axes[2].imshow(combined_image, cmap="gray")
    axes[2].set_title("Combined image", fontsize=10, pad=10) 
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"slice.png", bbox_inches='tight', dpi=100)
    logging.info(f"Saved figure slice.png")
    plt.close(fig)