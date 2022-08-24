import pathlib
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json


# Code taken from utils.py on https://github.com/alexsax/2D-3D-Semantics
def get_index(color):
    """Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    """
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def get_color(i):
    """Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    """
    b = (i) % 256  # least significant byte
    g = (i >> 8) % 256
    r = (i >> 16) % 256  # most significant byte
    return r, g, b


def load_semantic(filename, lbl_map):
    # Load the RBG image 24 bit base 256 encoded
    semantic_img = np.array(Image.open(filename))
    # Convert the RGB code into the index code

    semantic = np.zeros(semantic_img.shape[:2], dtype="int")
    for i in range(semantic_img.shape[0]):
        for j in range(semantic_img.shape[1]):
            semantic[i, j] = lbl_map[get_index(semantic_img[i, j, :])]
    return semantic


if __name__ == "__main__":
    rootdir = pathlib.Path("/opt/Datasets/stanford/")
    area = "area_3"
    rgbdir = rootdir / area / "data" / "rgb"
    semanticdir = rootdir / area / "data" / "semantic"
    semantic_json = rootdir / "assets" / "semantic_labels.json"

    # Load the semantic labels file
    with open(semantic_json) as f:
        json_labels = json.load(f)

    # Preprocess the labels to keep only the class names
    # lbl_map is a int -> int dictionnary mapping the original long list of labels
    # down to only the 14 classes
    # ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door', 'floor', 'sofa', 'table',
    #        'wall', 'window']
    labels = sorted(list(set([lblname.split("_")[0] for lblname in json_labels])))
    lbl_map = {ik: labels.index(k.split("_")[0]) for ik, k in enumerate(json_labels)}
    lbl_map[int(0x0D0D0D)] = labels.index(
        "<UNK>"
    )  # 0x0D0D0D is encoding missing labeling

    num_labels = len(labels)
    print(f"I loaded {num_labels} labels : {labels}")

    # Let's list all the rgb_files
    rgb_files = rgbdir.glob("*.png")

    # Pick a random image for testing our script
    img0_filename = next(iter(rgb_files))
    semantic_img0_filename = semanticdir / (
        str(img0_filename.name)[:-7] + "semantic.png"
    )

    # Load the RGB image a 1080x1080x3 image
    print(f"Loading {img0_filename}")
    rgb_img0 = np.array(Image.open(img0_filename))
    # Load the semantic labels image, a 1080x1080 image with
    # values in [0, num_labels-1]
    print(f"Loading {semantic_img0_filename}")
    semantic_img0 = load_semantic(semantic_img0_filename, lbl_map)

    print("In the current image, I found the classes :")
    print("{}".format(",".join(labels[cls] for cls in np.unique(semantic_img0))))

    # For display purpose, color the semantic image
    # i.e. map the 1080x1080 matrix of index onto a
    # 1080x1080 RGB image
    # We use the tab20 colormap but only 14 colors are really used
    colours = cm.get_cmap("tab20", 20)
    cmap = colours(np.arange(num_labels))
    colored_semantic = cmap[semantic_img0.flatten()].reshape(
        semantic_img0.shape + (-1,)
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb_img0)
    ax[0].set_axis_off()
    ax[1].imshow(colored_semantic)
    ax[1].set_axis_off()

    plt.savefig("stanford.png", bbox_inches="tight")
    plt.show()
