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

def build_image_dataset(root_dir, valid_ratio, *params):    
    dataset = ImageDataset(root_dir, *params)
    return dataset

class BaseImageDataset(Dataset):

    def __init__(self, root_dir, img_idx=0) -> None:
        super().__init__()

        # List all the images in root_dir
        self.images = []
        for ext in img_extensions:
            self.images.extend(glob.glob(f"{root_dir}/{ext}"))

        # And take ONE of these images for our dataset
        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir} with extensions {img_extensions}")
        logging.info(f"Using the image file {self.images[img_idx]} for the dataset")
        self.filename = self.images[img_idx]

        # Load image, normalize in [0, 1], square pad
        im = np.array(Image.open(self.filename)).astype(np.float32) / 255.0
        H, W, C = im.shape
        size = max(H, W)
        self.image = np.ones((size, size, C), dtype=np.float32)
        self.image[:H, :W, :] = im

        self.height = size
        self.width = size

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
        Return the pixel value at the given index. It snaps to pixel centers 
        for in-between pixels
        """

        # Fix our samplings to the integer coordinates
        # This does not work pretty well

        # height, width = self.image.shape[:2]
        # j = idx % width
        # i = idx // width
        # pixel_value = self.image[i, j]
        # return np.array([i, j], dtype=np.float32), pixel_value

        # Sample random continuous coordinates 
        # in [0, 1] range
        xs = np.random.rand(2).astype(np.float32)

        # Get value by piecewise constant sampling
        pixel_value = self.sample(xs)

        return xs, pixel_value.astype(np.float32)

    def __repr__(self):
        return f"ImageDataset(N={self.__len__()}, dim_input={self.dim_input}, dim_output={self.dim_output}, filename={self.filename})"

class ImageDataset(BaseImageDataset):

    def __repr__(self):
        return super().__repr__() + " Piecewise constant"

    def sample(self, xs: np.ndarray) -> np.ndarray:
        """
        Perform sampling by snapping floating point coordinates to their 
        nearest integers
        
        Arguments:
            xs: Array of shape (2,) with coordinates in [0, 1] range
                xs[0] corresponds to width (x), xs[1] corresponds to height (y)
            
        Returns:
            Interpolated pixel value
        """
        height, width = self.height, self.width

        ######################
        # START CODING HERE ##
        ######################
        # We perform sampling of the numpy array self.image which is (height, width, 3)
        # using nearest neighbour interpolation
        # @SOL
        x = int(np.floor(xs[0] * (width - 1.)))
        y = int(np.floor(xs[1] * (height - 1.)))

        return self.image[y, x]
        # SOL@
        # @TEMPL
        # return np.array([np.random.random(), np.random.random(), np.random.random()])
        # TEMPL@
        ####################
        # END CODING HERE ##
        ####################

class BilinearImageDataset(BaseImageDataset):
    """
    Image dataset that samples random continuous coordinates and returns
    bilinearly interpolated pixel values.
    
    Arguments:
        root_dir: Directory containing image files
    """

    def __repr__(self):
        return super().__repr__() + " Bilinear interpolation"

    def sample(self, xs: np.ndarray) -> np.ndarray:
        """
        Perform bilinear interpolation at continuous coordinates.
        
        Arguments:
            xs: Array of shape (2,) with coordinates in [0, 1] range
                xs[0] corresponds to width (x), xs[1] corresponds to height (y)
            
        Returns:
            Interpolated pixel value
        """
        height, width = self.height, self.width
        
        # Scale from [0, 1] to pixel coordinates
        # xs is in [0, 1], we scale to [0, width] and [0, height]
        x = xs[0] * width
        y = xs[1] * height
        
        # Get integer indices
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Compute interpolation weights
        lerp_x = x - x0
        lerp_y = y - y0
        
        # Clamp indices to valid range
        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)
        
        # Bilinear interpolation
        # See the illustration at :
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        # The corner pixel values are weights by 
        # the fraction of the area of the 
        # **opposite** quadrants.
        #
        #(y0,x0)            (y0, x1)
        #   +----|------------+
        #   |(1) |      (2)   |
        #   |    |            | 
        #   |-- (y,x)     ----|
        #   |    |            |
        #   |(3) |      (4)   |
        #   +-----------------+
        #(y1,x0)            (y1, x1) 

        # lerp_x = x - x0
        # lerp_y = y - y0
        # (1) weights (y1, x1)
        # (2) weights (y1, x0)
        # (3) weights (y0, x0)
        # (4) weights (y0, x0)

        # And : 
        # (1) = lerp_x * lerp_y
        # (2) = (1-lerp_x) * lerp_y
        # (3) = lerp_x * (1-lerp_y)
        # (4) = (1-lerp_x)*(1-lerp_y)
        # with (1) + (2) + (3) + (4) = 1

        ######################
        # START CODING HERE ##
        ######################
        # We perform sampling of the numpy array self.image which is (height, width, 3)
        # using bilinear interpolation
        # @SOL
        a1 = lerp_x * lerp_y
        a2 = (1.0-lerp_x)*lerp_y
        a3 = lerp_x * (1.0-lerp_y)
        a4 = (1.0-lerp_x)*(1.0-lerp_y)

        value = (
            self.image[y0, x0] * a4 +
            self.image[y0, x1] * a3 +
            self.image[y1, x0] * a2 +
            self.image[y1, x1] * a1
        )
        
        return value
        # SOL@
        # @TEMPL
        # return np.array([np.random.random(), np.random.random(), np.random.random()])
        # TEMPL@
        ####################
        # END CODING HERE ##
        ####################
