# coding: utf-8

# External imports
import torch

# Local imports

def build_model(cfg, input_size):
    return eval(f"{cfg['class']}(cfg, input_size)")
