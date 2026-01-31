# coding: utf-8

# Standard import
import logging
import sys

# External imports
import torch

from nirlab import utils
try:
    import tinycudann as tcnn
    tcnn_available = True
except ImportError:
    print("tiny-cuda-nn module not available")
    print("You will not be able to use the Hash encoding")
    tcnn_available = False

# Local imports
from . import build_model
from . import encoding

def test_real_NGP():
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

    K = 123
    X = torch.rand(K, dim_input)
    y = nir(X)
    
    assert( y.shape == torch.Size((K, dim_output)))
    
def test_real_hash_NGP():
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
    

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_real_NGP()
    if tcnn_available:
        test_real_hash_NGP()
