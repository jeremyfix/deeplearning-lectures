#!/usr/bin/env python3

"""
In this script, we experiment with various visualisation techniques for
demangling what is learned by a deep neural network

For saliency_simonyan
    Simonyan et al. (2014), Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps. https://arxiv.org/pdf/1312.6034.pdf

Additional references:
    Montavon et al. (2017) Methods for Interpreting and Understanding Deep Neural Networks, https://arxiv.org/pdf/1706.07979.pdf

"""


# Standard modules
import argparse
# External modules
import torch
import torchvision
from PIL import Image
# Local modules
import models


def saliency_simonyan(device, args):
    """
    Takes a pretrained model and an input image and computes the 
    saliency over this input image according to [Simonyan et al.(2014)]
    """
    # Checks we have the required arguments
    if not args.image:
        raise RuntimeError("I need an input image to work")

    # Loads a pretrained model
    image_transform, model = models.get_model("resnet50", device)

    # Switch model to eval mode as it may have evaluation specific
    # layers
    model.eval()

    # Loads the image
    img = Image.open(args.image).convert('RGB')

    # Go through the model
    input_tensor = image_transform(img).to(device).unsqueeze(0)
    out = model(input_tensor)


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--visu',
                        type=str,
                        choices=['saliency_simonyan'],
                        action='store',
                        required=True)

    parser.add_argument('--image',
                       type=str,
                       action='store',
                       help='The input image to process if required')

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using the GPU")
        device = torch.device('cuda')
    else:
        print("Using the CPU")
        device = torch.device('cpu')

    exec("{}(device, args)".format(args.visu))



