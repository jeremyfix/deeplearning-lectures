#!/usr/bin/env python3

"""
In this script, we experiment with various visualisation techniques for
demangling what is learned by a deep neural network

For saliency_simonyan
    Simonyan et al. (2014), Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps. https://arxiv.org/pdf/1312.6034.pdf
    Yosinski et al. (2015), Understanding Neural Networks Through Deep Visualization

Additional references:
    - Montavon et al. (2017) Methods for Interpreting and Understanding Deep Neural Networks, https://arxiv.org/pdf/1706.07979.pdf
    - https://github.com/utkuozbulak/pytorch-cnn-visualizations

"""

# Standard modules
import os
import argparse
import sys
import yaml
# External modules
from PIL import Image
import torch
import torchvision
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
# Local modules
import models
import utils


# For testing an image

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

def register_activation_hooks(dact, modules):
    """
    See : https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
    """
    def hook_fn(m, i, o):
        dact[m] = o

    for name, layer in modules:
        if isinstance(layer, (torch.nn.Sequential)):
            register_activation_hooks(dact, layer.modules())
        else:
            layer.register_forward_hook(hook_fn)


def get_activations(params, device, tensorboard_writer):
    """
    Forward propagate an image through a network
    and saves/displays the activations through the layers
    The activations are exported on the tensorboard.

    This function allows to see how to access the inner structure of the model
    """

    model = params['model']

    # Loads a pretrained model
    image_transforms, model = models.get_model(model, device)
    image_normalize, image_denormalize = image_transforms

    activities = {}
    register_activation_hooks(activities, model.modules())


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

    if 'activations' in config:
        params = {'model': config['model']}
        params.update(config['activations'])
        get_activations(params, device, tensorboard_writer)
    if 'generate_image' in config:
        params = {'model': config['model']}
        params.update(config['generate_image'])
        generate_image(params, device, tensorboard_writer)


# if __name__ == '__main__':
#     main()
