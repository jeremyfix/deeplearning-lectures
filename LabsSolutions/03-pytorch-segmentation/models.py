#!/usr/bin/env python3
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

# Standard imports
import sys
import logging

# External imports
import torch
import torch.nn as nn
import torchvision.models

available_models = ["fcn_resnet50"]


class TorchvisionModel(nn.Module):
    def __init__(self, modelname, input_size, num_classes, pretrained, in_channels):
        super().__init__()
        if pretrained:
            logging.info("Loading a pretrained model")
        else:
            logging.info("Loading a model with random init")
        exec(
            f"self.model = torchvision.models.segmentation.{modelname}(pretrained={pretrained}, pretrained_backbone={pretrained})"
        )
        old_conv1 = self.model.backbone.conv1
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.stride,
        )
        old_head = self.model.classifier
        last_conv = list(old_head.children())[-1]
        self.model.classifier = nn.Sequential(
            *list(old_head.children())[:-1],
            nn.Conv2d(
                in_channels=last_conv.in_channels,
                out_channels=num_classes,
                kernel_size=last_conv.kernel_size,
                stride=last_conv.stride,
                padding=last_conv.stride,
            ),
        )

    def forward(self, x):
        return self.model(x)["out"]


def fcn_resnet50(input_size, num_classes):
    return TorchvisionModel("fcn_resnet50", input_size, num_classes, True, 3)


def build_model(model_name, img_size, num_classes):
    if model_name not in available_models:
        raise RuntimeError(f"Unavailable model {model_name}")
    exec(f"m = {model_name}(img_size, num_classes)")
    return locals()["m"]


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    license = """
    models.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    logging.info(license)
    mname = ["fcn_resnet50"]

    for n in mname:
        m = build_model(n, (224, 224), 85)
        out = m(torch.zeros(2, 3, 224, 224))
        print(out.shape)
