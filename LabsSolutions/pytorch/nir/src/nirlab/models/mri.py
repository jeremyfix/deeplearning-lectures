# coding: utf-8

# Standard imports
import sys

# External imports
import torch
import torch.nn as nn

# Local imports
from nirlab.models import ngp

class MRINerf(nn.Module):
    """
    NERF model of CineJense

    Arguments:
        dim_input (int): Number of input coordinates (e.g. 3 for x, y, t)
        dim_output (int): Number of output values (e.g. 1 for a scalar field)
        cfg (dict): Configuration for the encoding and MLP
    """

    def __init__(self,
                 cfg: dict):
        super().__init__()

        self.image_model = ngp.RealNGP(cfg["image"])
        self.csm_model = ngp.RealNGP(cfg["csm"])

    def forward(self, X):
        """
        Forward pass of the MRI NERF model

        Arguments:
            X: Input coordinates of shape (K, 3) with (x, y, t)

        Returns:
            outputs: TBD
        """
        assert X.shape[0] == 1, "Batch size must be 1 for MRINerf"
        X = X.squeeze(0) # Remove the batch index, now X is (K, 3)

        pre_intensity = torch.view_as_complex(self.image_model(X))  # (K, ) 
        
        csm = self.csm_model(X).view(X.shape[0], -1, 2)   # (K, ncoils, 2)
        csm = torch.view_as_complex(csm)  # (K, ncoils)

        # Compute the RSS over the coils
        csm_norm = torch.abs(csm).sum(axis=-1) + 1e-12
        csm = csm / csm_norm.unsqueeze(-1) # (K, ncoils)

        return pre_intensity, csm