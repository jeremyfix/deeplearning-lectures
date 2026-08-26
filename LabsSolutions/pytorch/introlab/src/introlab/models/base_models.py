# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch.nn as nn


def Linear(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    # @TEMPL
    # # TODO: Implement a simple linear model
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # layers = []
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@
    # @SOL
    layers = [
        nn.Flatten(start_dim=1),
        nn.Linear(reduce(operator.mul, input_size, 1), num_classes),
    ]
    # SOL@

    return nn.Sequential(*layers)


def FFN(cfg, input_size, num_classes):
    """
    cfg: a dictionnary with possibly some parameters
    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    num_layers = cfg.get("num_layers", 1)
    num_hidden = cfg.get("num_hidden", 32)
    use_dropout = cfg.get("use_dropout", False)
    # @TEMPL
    # # TODO: Implement a simple linear model
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # layers = []
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@
    # @SOL
    if use_dropout:
        layers = [nn.Flatten(start_dim=1)]
        input_dim = reduce(operator.mul, input_size, 1)
        for i in range(num_layers):
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, num_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            input_dim = num_hidden
        layers.append(nn.Linear(num_hidden, num_classes))
    else:
        layers = [nn.Flatten(start_dim=1)]
        input_dim = reduce(operator.mul, input_size, 1)
        for i in range(num_layers):
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, num_hidden))
            layers.append(nn.ReLU())
            input_dim = num_hidden
        layers.append(nn.Linear(num_hidden, num_classes))
    # SOL@
    return nn.Sequential(*layers)
