# coding: utf-8

# This file originates from the torchcvnn python library
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

# External imports
import torch
from torch.utils.data import Dataset
import h5py  # Required because the data are matlab v7.3 files
import numpy as np


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

    Note:
        An example usage :

        .. code-block:: python

            import torchcvnn
            from torchcvnn.datasets.miccai2023 import MICCAI2023, CINEView, AccFactor

            def process_kspace(kspace, coil_idx, slice_idx, frame_idx):
                coil_kspace = kspace[:, :, coil_idx, slice_idx, frame_idx]
                mod_kspace = np.log(np.abs(coil_kspace) + 1e-9)

                img = kspace_to_image(coil_kspace)
                img = np.abs(img)
                img = img / img.max()

                return mod_kspace, img

            dataset = MICCAI2023(rootdir, view=CINEView.SAX, acc_factor=AccFactor.ACC8)
            subsampled_kspace, subsampled_mask, full_kspace = dataset[0]

            frame_idx = 5
            slice_idx = 0
            coil_idx = 9

            mod_full, img_full = process_kspace(full_kspace, coil_idx, slice_idx, frame_idx)
            mod_sub, img_sub = process_kspace(subsampled_kspace, coil_idx, slice_idx, frame_idx)

            # Plot the above magnitudes
            ...


        Displayed below is an example patient with the SAX view and acceleration of 8:

        .. figure:: ../assets/datasets/miccai2023_sax8.png
           :alt: Example patient from the MICCAI2023 dataset with both the full sampled and under sampled k-space and images
           :width: 100%
           :align: center

        Displayed below is an example patient with the LAX view and acceleration of 4:

        .. figure:: ../assets/datasets/miccai2023_lax4.png
           :alt: Example patient from the MICCAI2023 dataset with both the full sampled and under sampled k-space and images
           :width: 100%
           :align: center

        You can combine the coils using the root sum of squares
        to get a magnitude image (real valued) with all the
        coil contributions.


        Below are examples combining the coils for a given
        frame and slice, for LAX (top) and SAX (bottom). It uses
        the function :py:func:`torchcvnn.datasets.miccai2023.combine_coils_from_kspace`

        .. figure:: ../assets/datasets/miccai2023_combined_lax.png
           :alt: Example LAX, combining the coils
           :width: 50%
           :align: center

        .. figure:: ../assets/datasets/miccai2023_combined_sax.png
           :alt: Example SAX, combining the coils
           :width: 50%
           :align: center

    """

    def __init__(
        self,
        rootdir: str,
        view: CINEView = CINEView.SAX,
        acc_factor: AccFactor = AccFactor.ACC4,
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

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        """
        Returns the subsampled k-space data, the mask and the fully sampled k-space data
        """
        patient = self.patients[idx]

        subsampled_data = None
        subsampled_mask = None
        fullsampled_data = None

        # Load the subsampled data
        logging.info(f"Loading {patient / self.input_filename}")
        subsampled_data = load_matlab_file(
            patient / self.input_filename, self.subsampled_key
        ).transpose(3, 4, 2, 1, 0)
        subsampled_data = subsampled_data["real"] + 1j * subsampled_data["imag"]
        # (kx, ky, sc, sz, t) for multi-coil data
        # e.g. (246, 512, 10, 10, 12)

        logging.info(f"Loading {patient / self.mask_filename}")
        subsampled_mask = load_matlab_file(
            patient / self.mask_filename, self.mask_key
        ).transpose(0, 1)
        # (kx, ky)
        # e.g. (246, 512)

        logging.info(
            f"Loading {self.fullsampled_rootdir / patient.name / self.input_filename}"
        )
        fullsampled_data = load_matlab_file(
            self.fullsampled_rootdir / patient.name / self.input_filename,
            self.fullsampled_key,
        ).transpose(3, 4, 2, 1, 0)
        fullsampled_data = fullsampled_data["real"] + 1j * fullsampled_data["imag"]
        # kx, ky, sc, sz, t
        # e.g. (246, 512, 10, 10, 12)

        return subsampled_data, subsampled_mask, fullsampled_data
