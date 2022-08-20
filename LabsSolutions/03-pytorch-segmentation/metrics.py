#!/usr/bin/env python3
# coding : utf-8
"""
    This script belongs to the lab work on semantic segmenation of the
    deep learning lectures https://github.com/jeremyfix/deeplearning-lectures
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


# See also
# https://arxiv.org/pdf/1910.07655.pdf

# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F

available_losses = ["FocalLoss", "CrossEntropyLoss"]


def CrossEntropyLoss():
    return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    """
    Implementation of the Focal Loss :math:`\\frac{1}{N} \\sum_i -(1-p_{y_i} + \\epsilon)^\\gamma \\log(p_{y_i})`

    Args:
        gamma: :math:`\\gamma > 0` puts more focus on hard misclassified samples

    Shape:
        - predictions :math:`(B, C)` : the logits
        - targets :math`(B, )` : the target ids to predict
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.eps = 1e-10

    def forward(self, predictions, targets):
        """
        Arguments:
            predictions: (B, K, d1, d2, ..) of pre-activation
            targets : (B, d1, d2, ..) of class indices
        """
        B, num_classes = predictions.shape[:2]
        predictions = predictions.view(B, num_classes, -1)  # B, K, d1*d2*d3*..
        predictions = predictions.transpose(1, -1)  # B, d1*d2*d3*...., K
        predictions = predictions.reshape(-1, num_classes)

        logp = F.log_softmax(predictions, dim=1)
        logp_t = logp.gather(dim=1, index=targets.view(-1, 1))

        weight = torch.pow(1.0 - torch.exp(logp_t) + self.eps, self.gamma)
        loss = (-weight * logp_t).mean()

        return loss


def build_loss(loss_name):
    if loss_name not in available_losses:
        raise RuntimeError(f"Unavailable loss {loss_name}")
    exec(f"loss = {loss_name}()")
    return locals()["loss"]


if __name__ == "__main__":

    B, K, H, W = 10, 11, 12, 13
    predictions = torch.rand(B, K, H, W)
    targets = torch.randint(low=0, high=K, size=(B, H, W))

    losses = [FocalLoss(1.0)]
    for l in losses:
        value = l(predictions, targets)
        print(f"For {l} : {value}")
