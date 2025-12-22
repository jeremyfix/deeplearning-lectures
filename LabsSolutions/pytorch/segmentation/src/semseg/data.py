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

matplotlib.use("TkAgg")
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
            A.Normalize(mean=0, std=1),  # divides by 255.
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

    # We load the dataset used for training
    train_valid_dataset = torchvision.datasets.VOCSegmentation(
        root=root_dir,
        image_set="train",
        transforms=None,
        download=False,
    )

    # Split the data into a train and valid fold
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    # nb_valid = int(valid_ratio * len(train_valid_dataset))
    indices = list(range(len(train_valid_dataset)))
    random.shuffle(indices)
    train_indices = indices[:nb_train]
    valid_indices = indices[nb_train:]

    train_dataset = torch.utils.data.Subset(train_valid_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, valid_indices)

    # If we are asked to normalize the data, we compute the statistics on the
    # train fold only
    mean, std = 0.0, 1.0
    if normalize:
        mean, std = compute_mean_std(train_dataset, crop_size)

    # Build our transform/augmentation functions
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
    
    # Combine the transforms to build our final datasets
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
    # # We will add some augmentations here later on
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
            print(f"For sample {i}, {num_unlabeled} pixels")

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

def test_dataloaders(root_dir):
    config = {
            "root_dir": root_dir,
            "batch_size": 16,
            "normalize": True,
            "crop_size": 256,
            "valid_ratio": 0.2,
            "num_workers": 4
    }
    get_dataloaders(config, use_cuda=False)

if __name__ == "__main__":
    # @TEMPL
    # plot_samples("/mounts/datasets/datasets/Pascal-VOC2012")
    # test_dataloaders("/opt/datasets/Pascal-VOC2012")
    # TEMPL@
    # @SOL
    plot_samples("/opt/datasets/Pascal-VOC2012")
    test_dataloaders("/opt/datasets/Pascal-VOC2012")
    look_for_unlabeled("/opt/datasets/Pascal-VOC2012")
    # SOL@
