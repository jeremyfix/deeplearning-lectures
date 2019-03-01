import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os.path
import sys

# For possible models, see https://pytorch.org/docs/stable/torchvision/models.html
model_builder = {'resnet18'     : lambda:torchvision.models.resnet18(pretrained=True),
                 'resnet34'     : lambda:torchvision.models.resnet34(pretrained=True),
                 'resnet50'     : lambda:torchvision.models.resnet50(pretrained=True),
                 'resnet152'    : lambda:torchvision.models.resnet152(pretrained=True),
                 'densenet121'  : lambda:torchvision.models.densenet121(pretrained=True),
                 'squeezenet1_1': lambda:torchvision.models.squeezenet1_1(pretrained=True)}

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

class SingleBboxHead(nn.Module):

    def __init__(self, num_features: int, num_classes: int):
        super(SingleBboxHead, self).__init__()
        # We output four numbers. The semantics of these numbers
        # depend on the training set where we order the cy, cy, width, height
        # see data.py : targets_to_tensor
        self.head_bbox = nn.Sequential(nn.Dropout(), \
                                       nn.Linear(num_features, 128), nn.ReLU(), \
                                       nn.BatchNorm1d(128), \
                                       nn.Dropout(), \
                                       nn.Linear(128, 4), nn.Sigmoid())
        # We output the logits for all the classes. There is no "no-object class"
        self.head_class = nn.Sequential(nn.Linear(num_features, num_classes))

    def forward(self, features):
        # We might get feature maps as input
        # that we need to "linearize"
        features = features.view(features.size()[0], -1)

        # All the outputs for the bbox are projected into [0, 1]
        y_bbox  = self.head_bbox(features)
        y_class = self.head_class(features)

        return y_bbox, y_class

class MultipleBboxHead(nn.Module):
    def __init__(self, num_channels: int,
                 num_classes: int,
                 num_box: int):
        super(MultipleBboxHead, self).__init__()
        # The Bbox head outputs 4 x num_box  numbers per grid cell
        # i.e. every grid cell is predicting num_box bounding boxes
        self.head_bbox = nn.Sequential(\
                                       nn.Conv2d(num_channels, 1024, kernel_size=1,  stride=1, padding=0, bias=False), nn.ReLU(), \
                                       nn.BatchNorm2d(1024),\
                                       nn.Dropout(),\
                                       nn.Conv2d(1024, 512, kernel_size=1,  stride=1, padding=0, bias=False), nn.ReLU(), \
                                       nn.BatchNorm2d(512),\
                                       nn.Dropout(), \
                                       nn.Conv2d(512, 4*num_box, kernel_size=1,  stride=1, padding=0, bias=True),
                                       nn.Sigmoid())

        # The class head outputs a distribution over the classes
        # and a number which is the probability that the
        # cell contains an object
        self.head_class = nn.Sequential(\
                                       nn.Conv2d(num_channels, 1024, kernel_size=1,  stride=1, padding=0, bias=False), nn.ReLU(), \
                                       nn.BatchNorm2d(1024),\
                                       nn.Dropout(),\
                                       nn.Conv2d(1024, 512, kernel_size=1,  stride=1, padding=0, bias=False), nn.ReLU(), \
                                       nn.BatchNorm2d(512),\
                                       nn.Dropout(),\
                                       nn.Conv2d(512, num_box * (num_classes + 1), kernel_size=1,  stride=1, padding=0, bias=True)
        )

    def forward(self, features):

        # y_bbox outputs four numbers per cell cx, cy, width, height
        # we enforce cx, cy to lie in [0, 1]
        # width and height are relative to a grid cell size but can be
        # larger than 1.
        y_bbox  = self.head_bbox(features)

        # Reminder : y_class is of size batch_size x num_cells x num_cells x 21
        # 20 logits (class scores) + 1 logit (score for hosting an object)
        y_class = self.head_class(features)

        # In the return, we split the output with the
        # bbox_regression, class_logits, objectness_probability
        return y_bbox, y_class[:, :-1, :, :], y_class[:,-1, :, :]



class FeatureExtractor(nn.Module):

    def __init__(self, model_name:str):
        super(FeatureExtractor, self).__init__()
        model = model_builder[model_name]()

        # We freeze the networks
        # we will just perform forward passes, so we do not
        # need to compute any gradients
        # Does the following eliminate some computations ?
        for param in model.parameters():
            param.requires_grad = False

        # We keep only the feature maps of all the models
        # sometimes we can directly access a "features" attribute
        # sometimes we need to explictely extract the layers from the childrens
        if('resnet' in model_name):
            self.body = nn.Sequential(*list(model.children())[:-2])
        elif('densenet' in model_name):
            self.body = model.features
        elif('squeezenet' in model_name):
            self.body = model.features
        elif(model_name == 'mobilenetv2'):
            self.body = model

    def forward(self, x):
        return self.body(x)




if __name__ == '__main__':

    input_image_size = (250, 250)

    feature_extractor = FeatureExtractor(model_name='resnet18')

    # Build a tensor of 12 random RGB images
    x = torch.randn(12, 3, input_image_size[0], input_image_size[1])
    features = feature_extractor(x)

    # Let us build a multibox model
    multibox_model = MultipleBboxHead(features.shape[1], 1, 20)

    # From which we compute some bounding boxes and class labels
    y_bbox, y_class, y_obj = multibox_model(features)

