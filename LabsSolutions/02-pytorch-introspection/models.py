"""
Provides some utilitary functions to load a model with the right
image loader and preprocessing transform
"""

# Standard modules
# External modules
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

imagenet_preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

preprocessings = {
    'resnet18'     : imagenet_preprocessing,
    'resnet34'     : imagenet_preprocessing,
    'resnet50'     : imagenet_preprocessing,
    'resnet152'    : imagenet_preprocessing,
    'densenet121'  : imagenet_preprocessing,
    'squeezenet1_1': imagenet_preprocessing,
    'mobilenetv2'  : imagenet_preprocessing
}


def get_model(modelname, device):
    """
    Builds the model and image preprocessing function given the modelnalme

    Arguments:
        modelname : See the keys of model_builder
    Returns:
        image_transform_function, model
    """
    # Try to load a pretrained model
    try:
        model = model_builder[modelname]
    except KeyError:
        raise RuntimeError("Cannot build model named {}".format(modelname))
    # Send the model on the GPU if required
    model = model.to(device)

    # Now process the input image transform
    preprocessing = preprocessings[modelname]
    image_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          preprocessing])
    return image_transform, model
