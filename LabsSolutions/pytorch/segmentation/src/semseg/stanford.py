# coding: utf-8

# Standard imports
import pathlib
import json
import logging

# External imports
import PIL.Image as Image
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

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


