# coding: utf-8

# Standard imports
import logging
import glob
import sys

# External imports
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

img_extensions = ["*.jpg", "*.png", "*.ppm"]

class ImageDataset(Dataset):

    def __init__(self, root_dir) -> None:
        super().__init__()

        # List all the images in root_dir
        self.images = []
        for ext in img_extensions:
            self.images.extend(glob.glob(f"{root_dir}/{ext}"))

        # And take ONE of these images for our dataset
        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir} with extensions {img_extensions}")
        
        self.image = np.array(Image.open(self.images[0])).astype(np.float32)

    @property
    def dim_input(self) -> int:
        """
        Returns the dimension of the input, i.e. the 2 coordinates
        to index a pixel
        """
        return 2
    
    @property
    def dim_output(self) -> int:
        """
        Returns the dimension of the output, i.e. 3 for a RGB image
        and 1 for a grayscale image
        """
        return self.image.shape[-1]

    def __len__(self) -> int:
        """
        Return the size of the dataset. Here, the total number of pixels
        in the image
        """
        return self.image.shape[0] * self.image.shape[1]

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Return the pixel value at the given index
        """
        height, width = self.image.shape[:2]
        j = idx % width
        i = idx // width
        pixel_value = self.image[i, j]
        return np.array([i, j], dtype=np.float32), pixel_value


def test_image_dataset(rootdir):
    dataset = ImageDataset(rootdir)
    logging.info(f"Dataset input dimension: {dataset.dim_input}")
    logging.info(f"Dataset output dimension: {dataset.dim_output}")
    logging.info(f"Dataset size: {len(dataset)}")

    logging.info("Trying to index the datset...")
    for i in range(len(dataset)):
        pixel_value = dataset[i]
        logging.info(f"Pixel {i} value: {pixel_value}")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    rootdir = "."
    test_image_dataset(rootdir)
