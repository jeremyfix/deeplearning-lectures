# coding: utf-8

# Standard imports
import sys
import logging

# External imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    import tinycudann as tcnn
    tcnn_available = True
except ImportError:
    print("tiny-cuda-nn module not available")
    print("You will not be able to use the Hash encoding")
    tcnn_available = False

# Local imports
import nirlab.utils as utils

def build_encoder(dim_input, cfg):
    params = cfg["params"]
    return eval(f"{cfg['class']}(dim_input, params)")

class Positional(nn.Module):
    """
    This class implements the positional encoding as proposed in 
    the original NeRF paper Mildenhall et al.(2020)
    """

    def __init__(self, 
                 dim_input: int, 
                 cfg: dict): 
        super().__init__()
        
        self.dim_input = dim_input
        self.L = cfg["L"]

    def forward(self, X):
        """
        Given X as a collection of points on which to evaluate
        the embedding with X : (K, dim_input)

        We compute the positional encodings as a tensor of shape

        (K, dim_input * 2 * L)

        with cos/sin embeddings of L frequencies for each coordinate
        """
        # Compute the pulsations as 
        # 2**i pi  with i in [0, L-1]
        w = 2**(torch.arange(self.L, device=X.device)) * torch.pi # (L, )
        ww = torch.cat((w, w))

        # We then compute exp(i w c) for every coordinate
        # X is (K, n),  w is (2 * L, )
        # we broadcast with (K, n, 1) and (1, 1, 2 * L)
        # to get the (K, n, 2L) frequencies
        f = X[:, :, torch.newaxis] * ww[torch.newaxis, torch.newaxis, :]
        f[:, :, :self.L] = f[:, :, :self.L].cos()
        f[:, :, self.L:] = f[:, :, self.L:].sin()

        return torch.flatten(f, start_dim=1)

def test_positional_encoding():
    cfg = { "L" : 4}
    dim_input = 1
    # Uniform sampling of the volume [-1, 1]^dim_input
    npoints = 1000
    cfg = {"class": "Positional", "params": {"L": 4}}

    enc = build_encoder(dim_input=dim_input, cfg=cfg)

    # X = -1. + 2.0 * torch.rand(npoints, dim_input)
    # The reshape can be usefull when dim_input = 1 to introduce that
    # dimension
    X = torch.linspace(-1., 1., npoints).reshape(npoints, dim_input)

    print(X.shape)
    encodings = enc(X)
    print(encodings.shape)
   
    plt.figure()
    for n_enc in range(encodings.shape[1]):
        plt.plot(X[:, 0], encodings[:, n_enc], 
                 color="k", 
                 linewidth=3./(1+2*n_enc), 
                 label=f"L={n_enc}")
    # Show an x value and the encoding by blue dots on each curve
    x = 0.425
    enc_x = enc(torch.tensor([[x]]))
    for n_enc in range(encodings.shape[1]):
        plt.scatter(x, enc_x[0, n_enc], c='b', marker='o') 
    plt.vlines([x], ymin=-1, ymax=1)
    plt.legend()
    plt.show()

def TcnnHash(dim_input: int, cfg: dict):
    """
    This function wraps the tiny-cuda-nn implementation of the Hash
    encoding
    """
    assert tcnn_available
    return tcnn.Encoding(n_input_dims = dim_input,
                         encoding_config = cfg)

def test_hash_encoding():
    cfg = {
        "class": "Hash", 
        "params": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "fixed_point_pos": False
        }
    }

    dim_input = 2

    # Uniform sampling of the volume [-1, 1]^dim_input
    npoints = 100
    enc = build_encoder(dim_input=dim_input, cfg=cfg)
    xy = utils.build_coordinate_Nd(npoints, npoints)

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    encodings = enc(xy.to(device))
  
    xy = xy.detach().cpu()
    encodings = encodings.detach().cpu()

    logging.info(f"From points of shape {xy.shape}, {xy.dtype} the encoding produced an embedding of shape {encodings.shape}, {encodings.dtype}")

    plt.figure()
    plt.imshow(encodings[:, 4].reshape((npoints, npoints)))
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_positional_encoding()
    if tcnn_available:
        test_hash_encoding()

