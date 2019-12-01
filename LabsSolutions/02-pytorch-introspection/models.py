"""
Provides some utilitary functions to load a model with the right
image loader and preprocessing transform
"""

# Standard modules
# External modules
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


# For possible models, see https://pytorch.org/docs/stable/torchvision/models.html
model_builder = {'resnet18': lambda: torchvision.models.resnet18(pretrained=True),
                 'resnet34': lambda: torchvision.models.resnet34(pretrained=True),
                 'resnet50': lambda: torchvision.models.resnet50(pretrained=True),
                 'resnet152': lambda: torchvision.models.resnet152(pretrained=True),
                 'densenet121': lambda: torchvision.models.densenet121(pretrained=True),
                 'mobielnetv2': lambda: torchvision.models.mobilenet_v2(pretrained=True),
                 'squeezenet1_1': lambda: torchvision.models.squeezenet1_1(pretrained=True)}

imagenet_normalize = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                        ])

imagenet_denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                            std=[1./0.229, 1./0.224, 1./0.225])


preprocessings = {
    'resnet18': (imagenet_normalize, imagenet_denormalize),
    'resnet34': (imagenet_normalize, imagenet_denormalize),
    'resnet50': (imagenet_normalize, imagenet_denormalize),
    'resnet152': (imagenet_normalize, imagenet_denormalize),
    'densenet121': (imagenet_normalize, imagenet_denormalize),
    'squeezenet1_1': (imagenet_normalize, imagenet_denormalize),
    'mobilenetv2': (imagenet_normalize,  imagenet_denormalize)
}


def get_model(modelname, device):
    """
    Builds the model and image preprocessing function given the modelnalme

    Arguments:
        modelname : See the keys of model_builder
    Returns:
        image_transform_functions, model

        The image_transform_functions contain both the normalization and
        denormalization functions
    """
    # Try to load a pretrained model
    try:
        model = model_builder[modelname]()
    except KeyError:
        raise RuntimeError("Cannot build model named {}".format(modelname))
    # Send the model on the GPU if required
    model = model.to(device)

    # Now process the input image transform
    image_transforms = preprocessings[modelname]

    return image_transforms, model


def test_normalize():
    """
    Test function for computing the Normalize function
    denormalizing another Normalize function
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_normalize = transforms.Normalize(mean=mean,
                                       std=std)
    dmean = [-mui/stdi for (mui, stdi) in zip(mean, std)]
    dstd = [1.0/stdi for stdi in std]
    t_denormalize = transforms.Normalize(mean=dmean,
                                         std=dstd)

    inp = torch.rand((3, 23, 26))
    out = t_denormalize(t_normalize(inp))
    diff = (inp - out).abs().numpy()
    print(diff.max())
    print(torch.allclose(inp, out, 1e-5))
    

if __name__ == '__main__':
    test_normalize()
