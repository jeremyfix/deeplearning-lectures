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
from nirlab.models.hash_encoding import HashEmbedder

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
        the embedding with X : (N, dim_input)

        We compute the positional encodings as a tensor of shape

        (N, dim_input * 2 * L)

        with cos/sin embeddings of L frequencies for each coordinate
        """
        # Compute the pulsations as 
        # 2**i pi  with i in [0, L-1]
        w = 2**(torch.arange(self.L, device=X.device)) * torch.pi # (L, )
        ww = torch.cat((w, w))

        # We then compute exp(i w c) for every coordinate
        # X is (N, d),  w is (2 * L, )

        ######################
        # START CODING HERE ##
        ######################
        # 4 lines of code

        # Step 1
        # we can broadcast X as (N, d,     1) 
        #              and w as (1, 1, 2 * L)
        # so that the product of both will compute the 2^l \pi x_i for every
        # component i and frequency l
        # to get the (N, d, 2L) frequencies

        # Step 2
        # Once these terms are computed, it remains to apply the cos/sin functions on their
        # respective part of the third dimension

        # Step 3
        # Finally, we flatten the (N, d, 2L) as a (N, d * 2L) tensor

        f = X[:, :, torch.newaxis] * ww[torch.newaxis, torch.newaxis, :]
        f[:, :, :self.L] = f[:, :, :self.L].cos()
        f[:, :, self.L:] = f[:, :, self.L:].sin()

        return torch.flatten(f, start_dim=1)
        # SOL@
        # @TEMPL
        # return torch.zeros((X.shape[0], self.dim_input * 2.0 * self.L))
        # TEMPL@
        ####################
        # END CODING HERE ##
        ####################

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
    X = torch.linspace(0, 1., npoints).reshape(npoints, dim_input)

    print(X.shape)
    encodings = enc(X)
    print(encodings.shape)
   
    plt.figure()
    ncos = encodings.shape[1] // 2
    for n_enc in range(encodings.shape[1]):
        if n_enc < ncos:
            nl = n_enc
            color = "tab:red"
            label = f"cos(L={nl})"
            linewidth = 3./(1+2*nl)
        else:
            nl = n_enc - ncos
            color = "tab:blue"
            label = f"sin(L={nl})"
            linewidth = 3./(1+2*nl)
        plt.plot(X[:, 0], encodings[:, n_enc], 
                 color=color, 
                 linewidth=linewidth, 
                 label=label)
    # Show an x value and the encoding by blue dots on each curve
    x = 0.425
    enc_x = enc(torch.tensor([[x]]))
    for n_enc in range(encodings.shape[1]):
        plt.scatter(x, enc_x[0, n_enc], c='k', marker='x', zorder=10) 
    plt.vlines([x], ymin=-1, ymax=1, color='k')
    plt.legend()
    plt.savefig("positional_encoding.png")
    # plt.show()

def TcnnHash(dim_input: int, cfg: dict):
    """
    This function wraps the tiny-cuda-nn implementation of the Hash
    encoding
    """
    assert tcnn_available, "tiny-cuda-nn is not available"
    assert dim_input in [2, 3], "Only 2D and 3D are supported by TCNN"
    return tcnn.Encoding(n_input_dims = dim_input,
                         encoding_config = cfg)

def test_tcnn_hash_encoding():
    cfg = {
        "class": "TcnnHash",
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


# @SOL
class Hash(nn.Module):
    """
    This class implements the Hash encoding as proposed in 
    the instant NGP paper Müller et al.(2022)
    """

    def __init__(self, 
                 dim_input: int, 
                 cfg: dict): 
        super().__init__()
        
        assert dim_input in [2, 3], "Only 2D and 3D are supported by the Hash encoding"

        self.dim_input = dim_input
        self.n_levels = cfg["n_levels"] # L
        self.n_features_per_level = cfg["n_features_per_level"] # F
        self.log2_hashmap_size = cfg["log2_hashmap_size"]  # log_2(T)
        self.base_resolution = cfg["base_resolution"] # Nmin
        self.per_level_scale = cfg["per_level_scale"] # b

        # The unique large prime numbers considered in the paper for the hash function
        self.pi1 = 1
        self.pi2 = 2_654_435_761
        self.pi3 = 805_459_861

        # Define the parameters of the encoding
        self.lookup_tables = nn.ParameterList(nn.Parameter(torch.empty(2**self.log2_hashmap_size, self.n_features_per_level)) for _ in range(self.n_levels))

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialization of the hash table entries. 
        From the paper p 6.:
        "We initialize the hash table entries using the uniform distribution U (-1e-4, 1e-4)"
        """
        for table in self.lookup_tables:
            nn.init.uniform_(table, a=-1e-4, b=1e-4)

    def hash(self, x):
        a = x * self.pi1
        b = x * self.pi2
        c = x * self.pi3
        # Computes the hash as the xor, then modulo T
        h_x = torch.bitwise_xor(torch.bitwise_xor(a, b), c)
        # The modulo T is done with a bitmasking operation on the last log_2(T) bits
        h_x = torch.remainder(h_x, 2**self.log2_hashmap_size)

    def forward(self, X):
        """
        Given X as a collection of points on which to evaluate
        the embedding with X : (K, dim_input)

        We compute the positional encodings as a tensor of shape

        (K, L * F)

        where L x F is the number of levels times the number of features per level
        """     

        # Do Bilinear interpolation in 2d
        # Do Trilinear interpolation in 3d
        # Or better apply the recursive formula as given on https://en.wikipedia.org/wiki/Bilinear_interpolation#Computation
        # to generalize the bilinear/trilinear interpolation to higher dimensions if needed
        raise NotImplementedError("The Hash encoding is not implemented yet.")

def test_hash_encoding():
    cfg = {
        "class": "Hash",
        "params": {
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5,
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
# SOL@

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_positional_encoding()
    # if tcnn_available:
    #     test_tcnn_hash_encoding()
    # @SOL
    test_hash_encoding()
    # SOL@

