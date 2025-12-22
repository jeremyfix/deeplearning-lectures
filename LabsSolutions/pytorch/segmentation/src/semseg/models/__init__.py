# coding: utf-8

# External imports
import torch

# Local imports
from .unet import *
from .smp import *


def build_model(cfg, input_size, num_classes):
    # When you have only two classes, you can use a single output
    if num_classes == 2:
        num_classes = 1
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
