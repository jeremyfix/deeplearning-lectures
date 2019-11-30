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
import sys
# External modules
from PIL import Image
import torch
import torchvision
from torch.utils import tensorboard
# Local modules
import models
import utils


def test_model(device, args):
    """
    Test script showing how to go through a pretrained model
    given an image
    """

    modelname = 'resnet50'
    image_transform, model = models.get_model(modelname, device)

    # Do not forget to put model in eval mode
    model.eval()

    # Loads an image
    img = Image.open(args.image).convert('RGB')

    # Go through the model
    input_tensor = image_transform(img).to(device).unsqueeze(0)
    out = model(input_tensor)

    print("The provided image is of class {}".format(out.argmax()))


def saliency_simonyan(device, args):
    """
    Takes a pretrained model and an input image and computes the
    saliency over this input image according to [Simonyan et al.(2014)]
    """
    # Checks we have the required arguments
    if not args.image:
        raise RuntimeError("I need an input image to work")

    class_idx = 954  # Bananas
    nsteps = 100
    alpha = 1e-2
    modelname = 'resnet50'
    shape = (3, 224, 224)


    # Loads a pretrained model
    image_transform, model = models.get_model(modelname, device)

    # Switch model to eval mode as it may have evaluation specific
    # layers
    model.eval()

    # Loads the image
    img = Image.open(args.image).convert('RGB')

    # Go through the model
    input_tensor = image_transform(img).to(device).unsqueeze(0)
    out = model(input_tensor)

    # Let us start with a random image
    generated_image = torch.rand_like(input_tensor, requires_grad=True)
    generated_image = generated_image.to(device)
    f_loss = torch.nn.CrossEntropyLoss()

    for i in range(nsteps):
        sys.stdout.flush()
        loss = f_loss(model(generated_image), torch.LongTensor([class_idx]))
        loss.backward()
        # generated_image = generated_image + alpha * generated_image.grad
        generated_image.data.add_(alpha, generated_image.grad)
        sys.stdout.write("\r Step {}, Loss : {}".format(i, loss))


def main():
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


if __name__ == '__main__':
    main()
