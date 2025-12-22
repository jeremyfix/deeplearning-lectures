# coding: utf-8

# Standard imports

# External imports
import torch.nn as nn
import segmentation_models_pytorch as smp


def DeepLabV3Plus(cfg, input_size, num_classes):
    cin, _, _ = input_size
    return smp.DeepLabV3Plus(in_channels=cin, classes=num_classes, **cfg["parameters"])
