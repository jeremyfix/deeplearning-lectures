#!/usr/bin/env python3

"""
In this script, we experiment with various visualisation techniques for
demangling what is learned by a deep neural network

For saliency_simonyan
    Simonyan et al. (2014), Deep Inside Convolutional Networks: VisualisingImage Classification Models and Saliency Maps. https://arxiv.org/pdf/1312.6034.pdf

"""


# Standard modules
import argparse
# External modules
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--visu',
                        type=str,
                        choices=['saliency_simonyan'],
                        action='store',
                        required=True)

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load a model pretrained on ImageNet
    model = torchvision.models.resnet50(pretrained=True)
    model.to(device)
