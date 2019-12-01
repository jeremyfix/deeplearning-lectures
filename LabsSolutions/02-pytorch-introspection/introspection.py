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
from torch.utils.tensorboard import SummaryWriter
# Local modules
import models
import utils


def test_model(device, args):
    """
    Test script showing how to go through a pretrained model
    given an image
    """

    # Checks we have the required arguments
    if not args.image:
        raise RuntimeError("I need an input image to work")

    modelname = 'resnet50'
    image_transforms, model = models.get_model(modelname, device)
    image_normalize, image_denormalize = image_transforms

    # Do not forget to put model in eval mode
    model.eval()

    # Loads an image
    img = Image.open(args.image).convert('RGB')

    # Go through the model
    input_tensor = image_normalize(img).to(device).unsqueeze(0)
    out = model(input_tensor)

    print("The provided image is of class {}".format(out.argmax()))


def simonyan_generate_image(device, args):
    """
    Takes a pretrained model and an input image and computes the
    saliency over this input image according to [Simonyan et al.(2014)]
    """

    class_idx = 954  # Bananas
    nsteps = 1000
    alpha = 10
    modelname = 'resnet18'
    shape = (1, 3, 224, 224)

    # Tensorboard
    logdir = utils.generate_unique_logpath(args.logdir, "simonyan")
    tensorboard_writer = SummaryWriter(log_dir=logdir)

    # Loads a pretrained model
    image_transforms, model = models.get_model(modelname, device)
    image_normalize, image_denormalize = image_transforms

    # Switch model to eval mode as it may have evaluation specific
    # layers
    model.eval()

    # Generate an image that maximizes the probability
    # of being a member of class_idx

    # Let us start with a random image
    generated_image = torch.rand(shape, requires_grad=True)
    generated_image = generated_image.to(device)
    f_loss = torch.nn.CrossEntropyLoss()

    # Performs the gradient ascent
    for i in range(nsteps):
        sys.stdout.flush()
        logits = model(generated_image)
        loss = -f_loss(logits, torch.LongTensor([class_idx]))
        prob = torch.nn.functional.softmax(logits).squeeze()[class_idx]
        reg = generated_image.norm()
        total_loss = loss + reg
        total_loss.backward()
        # generated_image = generated_image + alpha * generated_image.grad
        generated_image.data.add_(alpha, generated_image.grad)
        sys.stdout.write("\r Step {}, Loss : {}, Prob : {}".format(i, loss, prob) + " "*50)
        tensorboard_writer.add_image("Generated image",
                                     image_denormalize(generated_image.squeeze()),
                                     i)


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--visu',
                        type=str,
                        choices=['simonyan_generate_image'],
                        action='store',
                        required=True)

    parser.add_argument('--image',
                        type=str,
                        action='store',
                        help='The input image to process if required')

    parser.add_argument('--logdir',
                        type=str,
                        default="./logs",
                        action='store')

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
