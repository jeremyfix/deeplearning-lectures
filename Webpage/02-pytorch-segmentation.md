---
title:  'Semantic segmentation'
author:
- Jeremy Fix
keywords: [PyTorch tutorial, semantic segmentation, Stanford 2D-3D S]
...

## Objectives

The objective of this lab work is to implement and explore convolutional neural networks for semantic segmentation. **Semantic segmentation** seeks to learn a function $f$, parametrized by $\theta$ which takes as input a colored image $I$ of arbitrary shape $H\times W$ and outputs an images of labels of the same shape $H \times W$ than the input. Indeed, we seek to label every single pixel of the image as belonging to one of $K$ predefined classes.

In this labwork, we will be working with the large [Stanford 2D-3D S dataset](http://buildingparser.stanford.edu/dataset.html). This dataset is built from 3D scans of buildings with multiple annotation types (pixelwise depth, pixel class, pixelwise normals, scene category). We will consider only the pixelwise class labeling. The data have been collected in 6 different areas. 

| Area | Number of images|
| ---  | --- |
| 1    | 10,327 |
| 2    | 5,382 |
| 3    | 1,530 |
| 4    | 4,644 |
| 5    | 6,714 |
| 6    | 3,744 |
| **Total** | 25,434 |

Below is an example of the input RGB image and the associated labels 
