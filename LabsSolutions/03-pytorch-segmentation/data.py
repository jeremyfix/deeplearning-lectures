#!/usr/bin/env python3
# coding : utf-8

# Standard imports
import sys
import pathlib
import json
import logging

# External imports
import torch
import numpy as np


class StanfordDataset:
    def __init__(self, rootdir: pathlib.Path):
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
        # Build a translation directory to map all the differently named labels
        # to the same ids;
        # For example, beam_10_hallway_6_1 and beam_10_storage_4_2 will be mapped
        # to the same label id
        self.lbl_map = {
            ik: self.labels.index(k.split("_")[0]) for ik, k in enumerate(json_labels)
        }
        self.lbl_map[int(0x0D0D0D)] = self.labels.index(
            "<UNK>"
        )  # 0x0D0D0D is encoding missing labeling

        self.num_labels = len(self.labels)
        logging.debug(f"I loaded {self.num_labels} labels : {self.labels}")

        # Look for the area directories
        self.filenames = {}
        for path in rootdir.iterdir():
            if path.is_dir():
                if path.name.startswith("area_"):
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

    def __getitem__(self, idx):
        """
        Args:
            idx : the index of the sample to return

        Returns
            (rgb, semantics, area_id) where
                rgb : (H, W, 3) source image
                semantics : (H, W, nclasses) labels
                area_id : int
        """
        pass


def get_dataloaders(
    rootdir: pathlib.Path,
    cuda: bool,
    batch_size: int,
    n_workers: int,
    small_experiment: bool,
    val_ratio: int,
):
    # Get the raw dataset
    dataset = StanfordDataset(rootdir)

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

    return train_loader, valid_loader


def test_dataloaders():
    rootdir = pathlib.Path("/opt/Datasets/stanford")
    cuda = False
    batch_size = 32
    n_workers = 4
    small_experiment = True
    val_ratio = 0.2

    train_loader, valid_loader = get_dataloaders(
        rootdir, cuda, batch_size, n_workers, small_experiment, val_ratio
    )
    logging.info(f"Train loader has {len(train_loader)} minibatches")
    logging.info(f"Valid loader has {len(valid_loader)} minibatches")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
    license = """
    data.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    logging.info(license)

    test_dataloaders()
