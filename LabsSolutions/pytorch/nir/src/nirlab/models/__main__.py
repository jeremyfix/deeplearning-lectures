# coding: utf-8

# Standard import
import logging
import sys

# External imports
from matplotlib.axis import XAxis
import torch
import torchinfo

# @SOL
try:
    import tinycudann as tcnn
    tcnn_available = True
except ImportError:
    print("tiny-cuda-nn module not available")
    print("You will not be able to use the Hash encoding")
    tcnn_available = False
# SOL@

# Local imports
from . import build_model
from . import encoding
from nirlab import utils

def test_real_NGP():
    logging.info("="*80)
    logging.info("Testing the real NGP with positional encoding")
    dim_input = 3
    dim_output = 4

    cfg = {
        "class": "RealNGP",
        "params": {
            "dim_input": dim_input,
            "dim_output": dim_output,
            "pos_encoding": {
                "class": "Positional",
                "params": {
                    "L": 10
                }
            },
            "n_hidden_units": 4,
            "n_hidden_layers": 5
        }
    }

    nir = build_model(cfg)
    logging.info(f"Summary of the model : \n{torchinfo.summary(nir, verbose=0)}\n\n")

    K = 123
    X = torch.rand(K, dim_input)
    y = nir(X)
    
    assert( y.shape == torch.Size((K, dim_output))), f"Got an output of size {y.shape}, expected {(K, dim_output)}"
   
# @SOL
def test_real_hash_NGP():
    logging.info("="*80)
    logging.info("Testing the NGP with Hash encoding")

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    dim_input = 2
    dim_output = 4

    cfg = {
        "class": "RealNGP",
        "params": {
            "dim_input": dim_input,
            "dim_output": dim_output,            
            "pos_encoding": {
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
            },
            "n_hidden_units": 4,
            "n_hidden_layers": 5
        }
    }

    nir = build_model(cfg).to(device)

    K = 123
    X = torch.rand(K, dim_input).to(device)
    XY = utils.build_coordinate_Nd(10, 10).to(device)
    y = nir(X)
    
    assert( y.shape == torch.Size((K, dim_output)))
# SOL@

def test_real_hashEmbedder_NGP():
    logging.info("="*80)
    logging.info("Testing the NGP with HashEmbedder encoding")

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')

    dim_input = 3
    dim_output = 4

    cfg = {
        "class": "RealNGP",
        "params": {
            "dim_input": dim_input,
            "dim_output": dim_output,            
            "pos_encoding": {
                "class": "HashEmbedder",
                "params": {
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "finest_resolution": 512
                }
            },
            "n_hidden_units": 4,
            "n_hidden_layers": 5
        }
    }

    nir = build_model(cfg).to(device)

    K = 123
    X = torch.rand(K, dim_input).to(device)
    y = nir(X)
    logging.info(f"Summary of the model : \n{torchinfo.summary(nir, verbose=0)}\n\n")
    
    assert( y.shape == torch.Size((K, dim_output))), f"Got an output of size {y.shape}, expected {(K, dim_output)}"


def test_MRINerf():
    logging.info("="*80)
    logging.info("Testing the MRINerf with Hash encoding")
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device('cpu')
    
    (kx, ky) = (204, 512)
    Nc = 10
    Nt = 12

    cfg = {
        "class": "MRINerf",
        "params": {
            "image": {
                "dim_input": 3, # x,y,t
                "dim_output": 2,
                "n_hidden_units": 32,
                "n_hidden_layers": 4,
                "pos_encoding": {
                    "class": "HashEmbedder",
                    "params": {
                        "n_levels": 14,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 17,
                        "base_resolution": 16,
                        "finest_resolution": 512
                    }
                }
            },
            "csm": {
                "dim_input": 3, # x,y,t
                "dim_output": 2*Nc,
                "n_hidden_units": 32,
                "n_hidden_layers": 4,
                "pos_encoding": {
                    "class": "HashEmbedder",
                    "params": {
                        "n_levels": 4,
                        "n_features_per_level": 8,
                        "log2_hashmap_size": 16,
                        "base_resolution": 2,
                        "finest_resolution": 64
                    }
                }
            }
        }
    }
    nir = build_model(cfg).to(device)
    logging.info(f"Summary of the model : \n{torchinfo.summary(nir, verbose=0)}\n\n")

    Nrows, Ncols, Nt = 12, 12, 8
    xyt = utils.build_coordinate_Nd(Nrows, Ncols, Nt).to(device).unsqueeze(dim=0)  # (K, 3)

    y = nir(xyt)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_real_NGP()
    # @SOL
    if tcnn_available:
        test_real_hash_NGP()
    # SOL@
    # @SOL
    test_real_hashEmbedder_NGP()
    test_MRINerf()
    # SOL@
    # @TEMPL
    # # test_real_hashEmbedder_NGP()
    # # test_MRINerf()
    # TEMPL@
