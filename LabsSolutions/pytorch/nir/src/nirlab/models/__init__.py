# coding: utf-8

# External imports
import torch

# Local imports
from .ngp import RealNGP
# @SOL
from .ngp import TinyCudaNGP, FullTinyCudaNGP
# SOL@
from .mri import MRINerf

def build_model(cfg):
    if "params" in cfg:
        params = cfg["params"]
        return eval(f"{cfg['class']}(params)")
    else:
        return eval(f"{cfg['class']}()")
    
