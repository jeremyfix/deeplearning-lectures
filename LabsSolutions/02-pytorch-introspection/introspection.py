#!/usr/bin/env python3

"""
In this script, we experiment with various visualisation techniques for
demangling what is learned by a deep neural network

For saliency_simonyan
    - Simonyan et al. (2014), Deep Inside Convolutional Networks: Visualising
                              Image Classification Models and Saliency Maps.
                              https://arxiv.org/pdf/1312.6034.pdf
    - Yosinski et al. (2015), Understanding Neural Networks Through Deep
                              Visualization

Additional references:
    - Montavon et al. (2017) Methods for Interpreting and Understanding Deep
                             Neural Networks,
                             https://arxiv.org/pdf/1706.07979.pdf
    - https://github.com/utkuozbulak/pytorch-cnn-visualizations

"""

# Standard modules
import os
import argparse
import sys
import collections
import functools
import math
# External modules
import yaml
from PIL import Image
import torch
import torchvision
import torchvision.models
import torchvision.utils
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


def list_modules(model):
    """
    List the modules and submodules of a model
    Example::

        >>> l = nn.Linear(2, 2)
        >>> m = nn.Sequential(l, l)
        >>> net = nn.Sequential(m, m)
        >>> list_modules(net)

    """
    for idx, m in enumerate(model.modules()):
        print(idx, ' ---> ', m)
        print('\n'*2)

def register_activation_hooks(dact, module):
    """
    See : https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
    """
    def hook_fn(name, m, i, o):
        print("Appending a value for {}".format(name))
        dact[name] = o

    for name, layer in module.named_modules():
        # print("Processing : {}, {}".format(name, layer))
        if isinstance(layer, (torch.nn.Conv2d)):
            print(" =====> {} Registered ".format(name))
            layer.register_forward_hook(functools.partial(hook_fn, name))


def get_activations(params, device, tensorboard_writer):
    """
    Forward propagate an image through a network
    and saves/displays the activations through the layers
    The activations are exported on the tensorboard.

    This function allows to see how to access the inner structure of the model
    """
    print("Get activations")
    model = params['model']

    # Loads a pretrained model
    image_transforms, model = models.get_model(model, device)
    image_normalize, _ = image_transforms

    # Container for the activities
    # If a key is missing, it is created with an empty list as value
    activities = collections.defaultdict(list)
    register_activation_hooks(activities, model)

    # Loads an image
    img = Image.open(params['image']).convert('RGB')
    # Go through the model
    input_tensor = image_normalize(img).to(device).unsqueeze(0)
    _ = model(input_tensor)

    # We can now register these activites on the tensorboard
    for k, v in activities.items():
        num_channels = v.size()[1]
        # Normalize all the activations to lie in [0, 1]
        for channel_idx in range(num_channels):
            tensor = v[0, channel_idx, :, :]
            tmax = tensor.max()
            tmin = tensor.min()
            if tmax != tmin:
                tensor[...] = (tmax - tensor)/(tmax - tmin)
            else:
                tensor[...] = 0
        # Make it a grid and display
        nrow = int(math.sqrt(num_channels))
        panned_images = torchvision.utils.make_grid(v.permute(1, 0, 2, 3), nrow=nrow)
        tensorboard_writer.add_image("Layer {}".format(k), panned_images, 0)
        
        
def generate_image(params, device, tensorboard_writer):
    """
    Takes a pretrained model and an input image and computes the
    saliency over this input image according to [Simonyan et al.(2014)]
    """
    class_idx = params['class_idx']
    nsteps = params['nsteps']
    lrate = params['optimizer']['lrate']
    momentum = params['optimizer']['momentum']
    l2reg = params['regularization']['l2']
    modelname = params['model']
    shape = (1, 3, 224, 224)

    # Loads a pretrained model
    image_transforms, model = models.get_model(modelname, device)
    image_normalize, image_denormalize = image_transforms

    # Switch model to eval mode as it may have evaluation specific
    # layers
    model.eval()

    # Generate an image that maximizes the probability
    # of being a member of class_idx

    # Let us start with a random image
    generated_image = torch.normal(mean=0, std=1,
                                   size=shape,
                                   requires_grad=True,
                                   device=device)

    # Instantiate the optimizer on the generated_image
    optimizer = torch.optim.SGD([generated_image],
                                lr=lrate, momentum=momentum,
                                nesterov=True)

    # Performs the gradient ascent
    for i in range(nsteps):
        sys.stdout.flush()
        logits = model(generated_image).squeeze()

        # Take the class score
        cls_score = logits[class_idx]
        # Computes the L2 of the image
        reg = generated_image.norm()

        # Computes the score loss with the regularizer
        total_loss = -cls_score + l2reg * reg

        # Backpropagates through it
        optimizer.zero_grad()
        total_loss.backward()

        # Update the generated image
        optimizer.step()

        # Computes the probability for display
        prob = torch.nn.functional.softmax(logits)[class_idx]

        # Debug display
        sys.stdout.write("\r Step {}, Score : {}, Prob : {}".format(i, cls_score, prob) + " "*50)
        tensorboard_writer.add_image("Generated image",
                                     image_denormalize(generated_image.squeeze()),
                                     i)
    sys.stdout.write('\n')


def main():

    # Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        type=str,
                        action='store',
                        required=True)

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError
    config = yaml.safe_load(open(args.config))

    print(config)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using the GPU")
        device = torch.device('cuda')
    else:
        print("Using the CPU")
        device = torch.device('cpu')

    # Tensorboard
    logdir = utils.generate_unique_logpath(config['logdir'], config['model'])
    tensorboard_writer = SummaryWriter(log_dir=logdir)

    image_transforms, model = models.get_model(config['model'], device)
    tensorboard_writer.add_graph(model, torch.zeros(1, 3, 224, 244))

    if 'activations' in config:
        params = {'model': config['model'],
                  'image': config['image']}
        params.update(config['activations'])
        get_activations(params, device, tensorboard_writer)
    if 'generate_image' in config:
        params = {'model': config['model'],
                  'image': config['image']}
        params.update(config['generate_image'])
        generate_image(params, device, tensorboard_writer)


if __name__ == '__main__':
    main()
