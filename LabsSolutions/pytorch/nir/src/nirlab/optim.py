# coding: utf-8

# External imports
import torch
import torch.nn as nn

# Local imports
from nirlab.utils import FFT

def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    return eval(f"torch.optim.{cfg['algo']}(params, **params_dict)")

class RelativeMSE(nn.Module):

    def forward(self, outputs, targets, epsilon=0.01):
        """
        Computes the relatise MSE

        In tiny-cuda-nn , they do the relative MSE normalization with 
        respect to outputs.detach()**2
        """
        relative_mse = (outputs - targets)**2 / (outputs.detach()**2 + epsilon)
        return relative_mse.mean()

class KSpaceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.reconstruction_loss = nn.HuberLoss()

    def forward(self, outputs, targets):
        """
        Compute the loss in the K space

        Both outputs and targets are expected to be Complex valued
        tensors
        """
        pre_intensity, csm = outputs 
        # pre_intensity : (K, )
        # csm : (K, ncoils)

        subsampled_data, subsampled_mask, fullsampled_data = targets

        # All the targets are expected to be of batch size = 1 because we process a full volume 
        assert subsampled_data.shape[0] == 1, "Batch size must be 1 for KSpaceLoss"
        subsampled_data = subsampled_data.squeeze(0)  # (kx, ky, sc, t)
        subsampled_mask = subsampled_mask.squeeze(0)  # (kx, ky, sc, t)
        fullsampled_data = fullsampled_data.squeeze(0)  # (kx, ky, sc, t)


        # subsampled_data (kx, ky, sc, t)
        Nrows, Ncols, Ncoils, Nframes = subsampled_data.shape
        pre_intensity = pre_intensity.view(Nrows, Ncols, Nframes, 1)  # (Nrows, Ncols, Nframes, 1)
        csm = csm.view(Nrows, Ncols, Nframes, Ncoils)  # (Nrows, Ncols, Nframes, Ncoils)

        # Apply the same pre-instensity through every coil specific sensitivity
        fft_pre_intensity = FFT(pre_intensity * csm).transpose(3, 2)  # (Nrows, Ncols, Ncoils, Nframes)
        
        # Compute the loss with the reconstruction loss
        # and the regularization loss
        observation_mask = subsampled_mask == 1

        masked_pred_kspace = torch.view_as_real(fft_pre_intensity[observation_mask])
        masked_kspace = torch.view_as_real(subsampled_data[observation_mask])

        loss_reconstruction = self.reconstruction_loss(masked_pred_kspace, masked_kspace)

        return loss_reconstruction

class TVLoss(nn.Module):
    """
    Total Variation Loss for 2D spatial smoothness.
    
    Samples random points in the normalized coordinate space
    and penalizes the difference between neighboring points.
    
    Arguments:
        model: The neural network model
        dim_input: Dimension of input coordinates (e.g. 2 for images)
        lbd: The weight/lambda for the TV loss
        N: Number of random points to sample
        delta (optional): The step size for computing finite differences (default 0.005=1./200.)
        coord_range (optional): Tuple of (min, max) for each dimension (default (0, 1)).
    """

    def __init__(self, 
                 model, 
                 lbd, 
                 N,
                 delta=0.005,
                 coord_bounds=(0, 1)):
        super().__init__()
        self.model = model
        self.dim_input = model.dim_input
        self.lbd = lbd
        self.delta = delta
        self.N = N
        
        self.coord_bounds = coord_bounds

    def forward(self):
        if self.lbd == 0:
            return torch.tensor(0.0)
        
        device = next(self.model.parameters()).device
        
        # Sample N random base points in coordinate range
        # Leave margin so shifted points stay in bounds
        coord_range = self.coord_bounds[1] - self.coord_bounds[0]
        X = self.coord_bounds[0] + self.delta + (coord_range - 2*self.delta) * torch.rand((self.N, self.dim_input), device=device) # N, dim_input

        # To compute a regular linspace, use the same function as build_coordinate_Nd
        # coords_lin = [torch.linspace(0, 1, self.N, device=device) for Ni in range(self.dim_input)]
        # coords_mesh = torch.meshgrid(*coords_lin, indexing="ij")
        # X = torch.stack(coords_mesh, -1).view(-1, self.dim_input)

        #  X[:, :, newaxis] +     shifts
        # (N, dim_input, 1) + (dim_input, N_shifts) 
        shifts = torch.eye(self.dim_input, device=device) # +delta for each direction, for each point
        X_dx = X[:, :, torch.newaxis] + self.delta * shifts # 
        # X_dx[i, :, j] is the point X[i, :] shifted by the j-th offset

        X_dx = X_dx.transpose(1, 2).reshape(-1, self.dim_input)
        # X_dx.permute(1, 2).reshape(-1, dim_input) is a vector
        # of shape (N*dim_input, dim_input)

        # Stack all points for a single forward pass: [X, X_dx]
        # Shape: (N+N*dim_input, dim_input)
        all_points = torch.cat([X, X_dx], dim=0)
        
        # Forward pass through model
        all_outputs = self.model(all_points)  # (N+N*dim_input, dim_output)
        
        # Split outputs
        y = all_outputs[:self.N]     # f(x)         (N, dim_output)
        y_dx = all_outputs[self.N:]  # f(x+shifts), (N*dim_input, dim_output)
      
        # Use broadcasting for duplicating the X points
        tv = (y[:, torch.newaxis, :] - y_dx).abs().mean()
        
        return self.lbd * tv

def test_tv_loss():
    dim_input = 2
    lbd = 0.1
    N = 5

    model = nn.Linear(dim_input, 1)
    tv = TVLoss(model, dim_input, lbd, N)
    print(tv())

if __name__ == "__main__":
    test_tv_loss()
