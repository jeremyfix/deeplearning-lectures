# coding: utf-8

# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    return eval(f"torch.optim.{cfg['algo']}(params, **params_dict)")

class RelativeMSE(nn.Module):

    def forward(self, outputs, targets):
        """
        Computes the relatise MSE as in tiny-cuda-nn
        code base

        Hey, they detached the normalization factor ?
        """
        relative_mse = (outputs - targets)**2 / (outputs.detach()**2 + 0.01)
        return relative_mse.mean()
