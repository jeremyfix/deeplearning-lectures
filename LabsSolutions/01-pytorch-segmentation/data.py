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


# Standard imports
import sys
import pathlib
import json
import logging
import random
import argparse

# External imports
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import tqdm

# Local imports
import utils


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rgb, semantics = self.dataset[index]
        return self.transforms(rgb, semantics)

    def __repr__(self):
        return f"{self.__class__.__name__} [\n -Dataset : {self.dataset} \n -Transforms : {self.transforms}\n]"


class StanfordDataset(torchvision.datasets.vision.VisionDataset):
    def __init__(
        self,
        rootdir: pathlib.Path,
        transforms=None,
        areas=None,
    ):
        super().__init__(rootdir, transforms, transform=None, target_transform=None)

        self.rootdir = rootdir

        semantic_json = rootdir / "assets" / "semantic_labels.json"
        if not semantic_json.exists():
            raise FileNotFoundError(f"File {semantic_json} does not exist")
        with open(semantic_json) as f:
            json_labels = json.load(f)
        # Preprocess the labels to keep only the class names
        # lbl_map is a int -> int dictionnary mapping the original long list of labels
        # down to only the 14 classes
        # ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table',
        #        'wall', 'window']

        # Extract the unique labels from ["<UNK>_0_<UNK>_0_0", "beam_10_hallway_6_1", "beam_10_storage_4_2", ...]
        self.labels = set([lblname.split("_")[0] for lblname in json_labels])
        # Keep the list sorted
        self.labels = sorted(list(self.labels))
        self.unknown_label = self.labels.index("<UNK>")
        # Build a translation directory to map all the differently named labels
        # to the same ids;
        # For example, beam_10_hallway_6_1 and beam_10_storage_4_2 will be mapped
        # to the same label id
        self.lbl_map = np.zeros((len(json_labels),), dtype=int)
        for ik, k in enumerate(json_labels):
            self.lbl_map[ik] = self.labels.index(k.split("_")[0])
        # This will be done while loading the semantics
        # self.lbl_map[int(0x0D0D0D)] = self.labels.index(
        #     "<UNK>"
        # )  # 0x0D0D0D is encoding missing labeling

        self.num_labels = len(self.labels)
        logging.debug(f"I loaded {self.num_labels} labels : {self.labels}")

        # Look for the area directories
        self.filenames = {}
        for path in rootdir.iterdir():
            if path.is_dir():
                if path.name.startswith("area_"):
                    area_number = str(path.name).split("_")[-1]
                    if areas is not None and area_number not in areas:
                        # Excluding this area
                        continue
                    area_name = path.name
                    rgb_path = path / "data" / "rgb"
                    img_paths = rgb_path.glob("*.png")
                    img_names = [p.name for p in img_paths]
                    # img_names are like ['camera_a024bdaf470f44d6af6813c3b119b38f_lounge_2_frame_18_domain_rgb.png', 'camera_fafa0629e8774618ac6e362d0416fba1_hallway_1_frame_14_domain_rgb.png',  ...]
                    self.filenames[area_name] = img_names
        logging.debug(f"I loaded {len(self.filenames)} areas")
        for area in self.filenames:
            logging.debug(f"Area {area} has {len(self.filenames[area])} images")

    def __len__(self):
        return sum(len(filenames) for _, filenames in self.filenames.items())

    def get_filename(self, idx):
        area_path = None
        for area_name, filenames_for_area in self.filenames.items():
            if idx < len(filenames_for_area):
                area_path = self.rootdir / area_name
                break
            # Otherwise decrement the index by the number of elements for this
            # area
            idx -= len(filenames_for_area)
        rgb_filename = self.filenames[area_name][idx]
        return rgb_filename, area_path

    def __getitem__(self, idx):
        """
        Args:
            idx : the index of the sample to return

        Returns
            (rgb, semantics, area_id) where
                rgb : (H, W, 3) PIL image
                semantics : (H, W) torch tensor of labels
                area_id : int
        """
        # Looking for the area in which the sample is
        rgb_filename, area_path = self.get_filename(idx)
        rgb_filepath = area_path / "data" / "rgb" / rgb_filename
        rgb_image = Image.open(rgb_filepath)

        # Load the semantic tensor
        semantic_filename = rgb_filename.replace("rgb", "semantic")
        semantic_filepath = area_path / "data" / "semantic" / semantic_filename
        semantic_img = np.array(Image.open(semantic_filepath))
        semantic_idx = (
            semantic_img[:, :, 0] * (256**2)
            + semantic_img[:, :, 1] * 256
            + semantic_img[:, :, 2]
        )
        # Replace the unlabeled pixels by UNK
        semantic_idx[semantic_idx == int(0x0D0D0D)] = self.unknown_label
        semantics = torch.from_numpy(
            self.lbl_map[semantic_idx].reshape(semantic_img.shape[:2])
        )
        return self.transforms(rgb_image, semantics)


def get_dataloaders(
    rootdir: pathlib.Path,
    cuda: bool,
    batch_size: int,
    n_workers: int,
    small_experiment: bool,
    val_ratio: int,
    train_transforms,
    valid_transforms,
    areas=None,
):
    # Get the raw dataset
    print(f"Transform is : {train_transforms}")
    dataset = StanfordDataset(
        rootdir, transforms=torchvision.datasets.vision.StandardTransform(), areas=areas
    )

    # Split it randomly in train/valid folds
    indices = list(range(len(dataset)))
    num_data = 128 if small_experiment else len(dataset)
    num_valid = int(val_ratio * num_data)
    num_train = num_data - num_valid

    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    valid_indices = indices[num_train : (num_train + num_valid)]

    # Build the train/valid datasets with the selected indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)

    train_dataset = TransformedDataset(train_dataset, train_transforms)
    valid_dataset = TransformedDataset(valid_dataset, valid_transforms)
    logging.info(f"Train dataset has {len(train_dataset)} samples")
    logging.info(f"Validation dataset has {len(valid_dataset)} samples")

    # And the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=cuda,
    )

    return train_loader, valid_loader, dataset.labels, dataset.unknown_label


# @SOL
def test_histogram(args):

    logging.info("Computing the histogram over the areas")

    # Build the original dataset
    dataset = StanfordDataset(
        args.datadir,
        transforms=torchvision.datasets.vision.StandardTransform(),
        areas=args.areas,
    )

    # Generate all the indices of the samples to just keep
    # a random fraction of them
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    num_samples = -1
    indices = indices[:num_samples]

    dataset = torch.utils.data.Subset(dataset, indices)

    # Iterate over the selected samples to get the statistics
    number_of_labels = torch.zeros((dataset.dataset.num_labels,))
    for _, mask in tqdm.tqdm(dataset):
        unique_values, counts = torch.unique(mask, return_counts=True)
        number_of_labels[unique_values] += counts
    number_of_labels /= number_of_labels.sum()
    print(number_of_labels)


# SOL@


def test_dataset(args):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    logging.info("Test dataset")

    to_tensor = transforms.ToTensor()
    data_transforms = lambda inp, targ: (to_tensor(inp), targ)
    # @SOL
    dataset = StanfordDataset(
        args.datadir,
        transforms=data_transforms,
        areas=args.areas,
    )
    data_idx = random.randint(0, len(dataset) - 1)  # 53899
    rgb, semantics = dataset[data_idx]
    # SOL@
    # @TEMPL
    # Code Here
    # vvvvvvvvv
    # dataset = ...
    # rgb, semantics = ...
    # ^^^^^^^^^
    # TEMPL@

    fig, axes = plt.subplots(1, 3, figsize=(6, 3))
    ax = axes[0]
    ax.imshow(rgb.permute(1, 2, 0).numpy())
    ax.set_title("Input image")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(utils.colorize(semantics.numpy()))
    ax.set_title("Semantics")
    ax.axis("off")

    colormap = utils.colormap.copy()
    colormap = list(colormap.items())
    # Sort the colormap by label name so that their display order
    # along the colorbar matches the label index of the dataset
    sorted(colormap, key=lambda lblcol: lblcol[0])

    # Preprend the unknown class which is labeled 0
    colormap.insert(0, ("unknown", [0, 0, 0]))

    ax = axes[2]
    cell_width = 212
    cell_height = 22
    patch_width = 48
    patch_height = 10
    for ilabel, (label, color) in enumerate(colormap):
        text_pos_x = patch_width + 7
        y = cell_height * ilabel
        ax.add_patch(
            Rectangle(
                xy=(0, y - patch_height / 2.0),
                width=patch_width,
                height=patch_height,
                facecolor=[2 * c / 255.0 for c in color],
                edgecolor="0.7",
            )
        )
        ax.text(
            text_pos_x,
            y,
            label,
            fontsize=10,
            horizontalalignment="left",
            verticalalignment="center",
        )
    ax.set_xlim(0, cell_width)
    ax.set_ylim(cell_height * (len(colormap) - 0.5), -cell_height / 2.0)
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def test_augmented_dataset(args):
    import matplotlib.pyplot as plt

    logging.info("Test augmented dataset")

    to_tensor = transforms.ToTensor()
    data_transforms = lambda inp, targ: (to_tensor(inp), targ)
    dataset = StanfordDataset(
        args.datadir, transforms=data_transforms, areas=args.areas
    )
    data_idx = random.randint(0, len(dataset) - 1)
    rgb_filename, area_path = dataset.get_filename(data_idx)
    logging.info("Loading the image " + str(area_path / "data" / "rgb" / rgb_filename))
    rgb, semantics = dataset[data_idx]

    print(f"The input image has type {rgb.dtype}, and shape {rgb.shape}")
    print(
        f"The semantic mask as type {semantics.dtype} and shape {semantics.shape}, with values in {semantics.unique()}"
    )

    # You can experiment your transforms
    # with the following code
    # by plugging into the pipeline, the
    # albumentations transform you think are relevant
    def data_transforms(img, mask):
        tf = A.Compose(
            [
                # @SOL
                A.RandomCrop(768, 768),
                A.Resize(256, 256),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(p=0.2),
                # A.CoarseDropout(max_width=50, max_height=50),
                A.MaskDropout((10, 15), p=1),
                # SOL@
                A.Normalize(0, 1),
                ToTensorV2(),
            ]
        )
        aug = tf(image=np.array(img), mask=mask.numpy())
        return (aug["image"], aug["mask"])

    dataset = StanfordDataset(
        args.datadir, transforms=data_transforms, areas=args.areas
    )
    aug_rgb, aug_semantics = dataset[data_idx]

    fig, axes = plt.subplots(1, 3)
    ax = axes[0]
    ax.imshow(utils.overlay(rgb.permute(1, 2, 0).numpy(), semantics.numpy()))
    ax.set_title("Original image")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(aug_rgb.permute(1, 2, 0).numpy())
    ax.set_title("Augmented image")
    ax.axis("off")

    ax = axes[2]
    ax.imshow(aug_rgb.permute(1, 2, 0).numpy())
    ax.imshow(utils.overlay(aug_rgb.permute(1, 2, 0).numpy(), aug_semantics.numpy()))
    ax.set_title("Augmented image overlayed with labels")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


# @SOL
def test_dataloaders(args):
    logging.info("Test dataloaders")

    cuda = False
    batch_size = 32
    n_workers = 4
    small_experiment = False
    val_ratio = 0.2

    # Note: ToTensor converts B,H,W,C to B, C, H, W and
    # maps [0, 255] to [0.0, 1.0]
    train_aug = A.Compose(
        [
            A.RandomCrop(768, 768),
            A.Resize(256, 256),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ]
    )
    valid_aug = A.Compose(
        [
            A.RandomCrop(768, 768),
            A.Resize(256, 256),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ]
    )

    def train_transforms(img, mask):
        aug = train_aug(image=np.array(img), mask=mask.numpy())
        return (aug["image"], aug["mask"])

    def valid_transforms(img, mask):
        aug = valid_aug(image=np.array(img), mask=mask.numpy())
        return (aug["image"], aug["mask"])

    train_loader, valid_loader, labels, _ = get_dataloaders(
        args.datadir,
        cuda,
        batch_size,
        n_workers,
        small_experiment,
        val_ratio,
        train_transforms,
        valid_transforms,
    )
    logging.info(f"Train loader has {len(train_loader)} minibatches")
    logging.info(f"Valid loader has {len(valid_loader)} minibatches")

    logging.info("Loading a minibatch from the training set")
    train_rgb, train_semantics = next(iter(train_loader))

    logging.info(f"The rgb images tensor has shape {train_rgb.shape}")
    logging.info(f"The semantic images tensor has shape {train_semantics.shape}")


# SOL@

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    license = """
    data.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    logging.info(license)

    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--areas",
        nargs="+",
        help="Which areas to use, specify it with their numbers (1, 2, 3, 4, 5a, 5b, 6)",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # test_histogram(args) @SOL@
    test_dataset(args)
    test_augmented_dataset(args)
    # test_dataloaders()  # @SOL@
