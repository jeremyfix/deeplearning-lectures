# coding: utf-8

# Standard import
import logging
import sys

# External imports
import torch

# Local imports
from . import build_model

def test_real_NGP():
    cfg = {
        "class": "RealNGP",
        "params": {
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

    dim_input = 3
    dim_output = 4

    nir = build_model(dim_input, dim_output, cfg)

    K = 123
    X = torch.rand(K, dim_input)
    y = nir(X)
    
    assert( y.shape == torch.Size((K, dim_output)))
    

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_real_NGP()
