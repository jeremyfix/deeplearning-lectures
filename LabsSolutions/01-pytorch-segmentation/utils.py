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

import numpy as np

colormap = {
    "beam": [95, 58, 51],
    "board": [32, 43, 45],
    "bookcase": [88, 18, 22],
    "ceiling": [92, 90, 42],
    "chair": [16, 19, 40],
    "clutter": [91, 91, 90],
    "column": [31, 64, 58],
    "door": [42, 53, 29],
    "floor": [36, 62, 77],
    "sofa": [34, 18, 37],
    "table": [31, 31, 30],
    "wall": [71, 45, 32],
    "window": [29, 68, 32],
}


def colorize(target):
    """
    Arguments:
        target: a numpy array of class indices (H, W)

    Returns:
        colored : a numpy array of colors (H, W, 3)
    """
    colored = np.zeros(target.shape + (3,), dtype="uint8")
    for icls, cls in enumerate(sorted(list(colormap.keys()))):
        color = colormap[cls]
        colored[target == (icls + 1)] = color
    # Double the colors to make them lighter
    # the max value of the colormap being 95
    return 2 * colored


def overlay(rgb, targets):
    """
    Overlay the semantic prediction on the input image
    rgb expected range in [0, 1], shape (H, W, 3)
    targets : nd array of predicted labels, shape (H, W)
    """
    colored_semantics = colorize(targets)
    lbd = 0.4
    ovlay = lbd * rgb + (1 - lbd) * colored_semantics / 255.0
    return ovlay
