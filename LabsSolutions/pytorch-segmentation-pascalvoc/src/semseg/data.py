# coding: utf-8

# Standard imports
import operator
import functools
import random
import logging

# External imports
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks
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


def compute_mean_std(dataset, crop_size, batch_size=128, num_workers=4):
    tfs = A.Compose(
        [
            A.SmallestMaxSize(max_size=crop_size),
            A.RandomCrop(height=crop_size, width=crop_size),
            A.Normalize(mean=0, std=1),  # divise par 255.
            ToTensorV2(),
        ]
    )

    dataset = WrappedDataset(dataset, tfs)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Compute the mean over minibatches of the
    # provided dataset
    mean = 0.0
    nsamples = 0.0
    for imgs, _ in loader:
        mean += imgs.sum()
        nsamples += functools.reduce(operator.mul, imgs.shape)
    mean /= nsamples

    # Compute the std over minibatches
    std = 0
    for imgs, _ in loader:
        std += ((imgs - mean) ** 2).sum()
    std /= nsamples
    std = torch.sqrt(std)

    return mean.item(), std.item()


def get_dataloaders(config: dict, use_cuda):

    root_dir = config["root_dir"]
    batch_size = config["batch_size"]
    normalize = config["normalize"]
    crop_size = config["crop_size"]
    valid_ratio = config["valid_ratio"]
    num_workers = config["num_workers"]

    # On charge l'ensemble des données qui vont servir à l'entrainement
    train_valid_dataset = torchvision.datasets.VOCSegmentation(
        root=root_dir,
        image_set="train",
        transforms=None,
        download=False,
    )

    # On découpe les données en un pli d'entrainement et un pli de validation
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    # nb_valid = int(valid_ratio * len(train_valid_dataset))
    indices = list(range(len(train_valid_dataset)))
    random.shuffle(indices)
    train_indices = indices[:nb_train]
    valid_indices = indices[nb_train:]

    train_dataset = torch.utils.data.Subset(train_valid_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, valid_indices)

    # Si on normalise les données, on calcule les statistiques de normalisation
    # sur les données d'entrainement uniquement
    mean, std = 0.0, 1.0
    if normalize:
        mean, std = compute_mean_std(train_dataset, crop_size)

    # On construit nos fonctions de transformation/augmentation
    augmentation_transforms = [
        # A.CoarseDropout(p=0.5, max_width=16, max_height=16),
        # A.PixelDropout(p=0.5),
        # A.MaskDropout(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ]
    resize_transforms = [
        A.SmallestMaxSize(max_size=crop_size),
        A.RandomCrop(height=crop_size, width=crop_size),
    ]
    logging.info(f"Using normalization : mean={mean}, std={std}")
    normalize_transforms = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    # On combine ces transformées pour construire nos datasets finaux
    train_transforms = A.Compose(
        augmentation_transforms + resize_transforms + normalize_transforms
    )
    train_dataset = WrappedDataset(train_dataset, train_transforms)

    valid_transforms = A.Compose(resize_transforms + normalize_transforms)
    valid_dataset = WrappedDataset(valid_dataset, valid_transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    num_classes = 21
    input_size = tuple(train_dataset[0][0].shape)

    return (
        train_loader,
        valid_loader,
        input_size,
        num_classes,
        {"mean": mean, "std": std},
        train_transforms,
    )


def process_mask(mask: torch.Tensor):
    """
    mask: a long tensor of shape [1, H, W] containing the class labels per pixel

    Returns:
        a [K, H, W] boolean tensor where each slice is class specific
    """
    _, H, W = mask.shape
    mask = mask.squeeze()
    num_classes = 21
    new_mask = torch.zeros(num_classes, H, W, dtype=torch.bool)
    for k in range(num_classes):
        new_mask[k][mask == k] = True
    return new_mask


def colorize(colormap, target):
    """
    Arguments:
        target: a numpy array of class indices (H, W)

    Returns:
        colored : a numpy array of colors (H, W, 3)
    """
    colored = np.zeros(target.shape + (3,), dtype="uint8")
    for icls, color in enumerate(colormap):
        colored[target == icls] = color
    return colored


def overlay(colormap, rgb, targets):
    """
    Overlay the semantic prediction on the input image
    rgb expected range in [0, 1], shape (H, W, 3)
    targets : nd array of predicted labels, shape (H, W)
    """
    colored_semantics = colorize(colormap, targets)
    lbd = 0.4
    ovlay = lbd * rgb + (1 - lbd) * colored_semantics / 255.0
    return ovlay


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def plot_samples(root_dir):
    nrows = 4
    ncols = 4

    crop_size = 256

    # @SOL
    augmentation_transforms = [
        A.CoarseDropout(p=0.5, max_width=16, max_height=16),
        A.PixelDropout(p=0.5),
        A.MaskDropout(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ]
    # SOL@
    # @TEMPL
    # # On ajoutera des augmentations ici
    # augmentation_transforms = []
    # TEMPL@
    resize_transforms = [
        A.SmallestMaxSize(max_size=crop_size),
        A.RandomCrop(height=crop_size, width=crop_size),
    ]
    conversion_transforms = [
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]

    data_transforms = A.Compose(
        augmentation_transforms + resize_transforms + conversion_transforms
    )

    dataset = torchvision.datasets.VOCSegmentation(
        root=root_dir,
        image_set="train",
        transforms=None,
        download=False,
    )

    dataset = WrappedDataset(dataset, data_transforms)

    # Les images ne sont pas toutes de même résolution, vous pouvez le voir avec le code ci-dessous
    # for i in range(len(dataset)):
    #    X, y = dataset[i]
    #    print(X.shape)

    # @SOL
    X, y = dataset[0]
    print(X.shape, y.shape)
    # SOL@
    # @TEMPL
    # # vvvvvvvvv
    # # CODE HERE
    # # Récuperez un échantillon annoté du dataset
    # # Quels sont les types et dimensions des tenseurs d'entrée et de sortie ?
    # # ^^^^^^^^^
    # TEMPL@

    fig, axes = plt.subplots(figsize=(10, 10), facecolor="w", nrows=nrows, ncols=ncols)

    # Attention, PascalVOC comprends une classe particulière, la 255
    # qui correspond à la délimitation des objets, elle sera représentée en blanc
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


# @SOL


def look_for_unlabeled(root_dir):
    resize_transforms = []
    conversion_transforms = [
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]

    data_transforms = A.Compose(resize_transforms + conversion_transforms)
    dataset = torchvision.datasets.VOCSegmentation(
        root=root_dir,
        image_set="train",
        transforms=None,
        download=False,
    )
    dataset = WrappedDataset(dataset, data_transforms)

    cmap = color_map()

    for i in range(len(dataset)):
        imgi, maski = dataset[i]
        num_unlabeled = (maski == 255).sum()
        if num_unlabeled >= 30000:
            print(num_unlabeled)

            imgi = imgi.squeeze().permute(1, 2, 0)  # (1, C, H, W) -> (H, W, C)
            maski = maski.squeeze()  # 1, 1, H, W -> H, W
            overlaid = overlay(cmap, imgi, maski)

            fig, axes = plt.subplots(figsize=(10, 10), facecolor="w", nrows=1, ncols=2)
            axes = axes.ravel()

            axes[0].imshow(imgi)
            axes[0].get_xaxis().set_visible(False)
            axes[0].get_yaxis().set_visible(False)

            axes[1].imshow(overlaid)
            axes[1].get_xaxis().set_visible(False)
            axes[1].get_yaxis().set_visible(False)
            plt.tight_layout()
            plt.show()


# SOL@

if __name__ == "__main__":
    # get_trainvalid_dataloaders("/mounts/Datasets4/Pascal-VOC2012", 64, True, 0.2, 7)
    # plot_samples("/mounts/Datasets4/Pascal-VOC2012") @TEMPL@
    # plot_samples("/opt/Datasets/Pascal-VOC2012")  # @SOL@
    look_for_unlabeled("/opt/Datasets/Pascal-VOC2012")  # @SOL@
