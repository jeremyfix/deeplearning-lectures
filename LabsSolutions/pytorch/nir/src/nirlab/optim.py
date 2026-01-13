# coding: utf-8

# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    return eval(f"torch.optim.{cfg['algo']}(params, **params_dict)")
