# coding: utf-8

# Standard imports
import logging
from PIL import Image
import glob

# External imports
from torch.utils.data import Dataset

img_extensions = ["*.jpg", "*.png", "*.ppm"]

class Image(Dataset):

    def __init__(self, root_dir):
        super().__init__()

        # List all the images in root_dir
        for infile in glob.glob("*.jpg"):
            print(infile)

        # And take ONE of these images for our dataset

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def test_dataset(rootdir):
    dataset = Image(rootdir)

if __name__ == "__main__":
    rootdir = "."
    test_image_dataset(rootdir)
