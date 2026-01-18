# coding: utf-8

# Standard imports
import operator
import functools
import random
import logging
import sys

# External imports
import torch
import numpy as np
import tqdm
import PIL
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WrappedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, tfs):
        super().__init__()
        self.dataset = dataset
        self.transforms = tfs

    def __getitem__(self, idx):
        X, mask = self.dataset[idx]
        if isinstance(X, PIL.Image.Image):
            X = np.array(X)
        if isinstance(mask, PIL.Image.Image):
            mask = np.array(mask)
        transformed_data = self.transforms(image=X, mask=mask)
        return transformed_data["image"], transformed_data["mask"]

    def __repr__(self):
        return f"{self.__class__.__name__}(dataset={self.dataset}, transform={self.transforms})"

    def __len__(self):
        return len(self.dataset)





def plot_samples(root_dir):
    nrows = 4
    ncols = 4

    dataset = MRIDataset(
        root=root_dir,
        image_set="train",
        transforms=None,
        download=False,
    )

    dataset = WrappedDataset(dataset, data_transforms)

    # The images do not all have the same size, you can check this with the code below
    # for i in range(len(dataset)):
    #    X, y = dataset[i]
    #    print(X.shape)

    # vvvvvvvvv
    # CODE HERE
    # Get an annotated sample from the dataset
    # What are the types and dimensions of the input/output tensors ?
    # @SOL
    X, y = dataset[0]
    print(X.shape, y.shape)
    # SOL@
    # # ^^^^^^^^^

    fig, axes = plt.subplots(figsize=(10, 10), facecolor="w", nrows=nrows, ncols=ncols)

    # Important : PascalVOC has a particular label, 255, which corresponds to the border
    # of the objects, which we represent in white
    cmap = color_map()

    for i, axi in enumerate(axes.ravel()):
        imgi, maski = dataset[i]
        imgi = imgi.squeeze().permute(1, 2, 0)  # (1, C, H, W) -> (H, W, C)

        maski = maski.squeeze()  # 1, 1, H, W -> H, W

        overlaid = overlay(cmap, imgi, maski)

        axi.imshow(overlaid)
        axi.get_xaxis().set_visible(False)
        axi.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig("pascalVOC_samples.png", bbox_inches="tight")
    print("Samples saved into pascalVOC_samples.png")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # @TEMPL
    # plot_samples("/mounts/datasets/datasets/Pascal-VOC2012")
    # test_dataloaders("/opt/datasets/Pascal-VOC2012")
    # TEMPL@
    # @SOL
    root_dir = "/opt/datasets/Pascal-VOC2012"
    # plot_samples(root_dir)
    # test_dataloaders(root_dir)
    # look_for_unlabeled(root_dir)
    class_stats(root_dir)
    # SOL@
