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


class GeneralizedDiceLoss:
    """
    Generalized Dice Loss as proposed by Sudre et al.(2017)

    C.H. Sudre, W. Li, T. Vercauteren, S. Ourselin, M.J. Cardoso (2017)
    Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations
    """

    def __init__(self):
        pass
