# coding: utf-8

# External imports
import torch

# Local imports
from .ngp import RealNGP, TinyCudaNGP, FullTinyCudaNGP

def build_model(dim_input, dim_output, cfg):
    if "params" in cfg:
        params = cfg["params"]
        return eval(f"{cfg['class']}(dim_input, dim_output, params)")
    else:
        return eval(f"{cfg['class']}(dim_input, dim_output)")
    
