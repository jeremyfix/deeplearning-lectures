# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


# @SOL


def VanillaCNN(cfg, input_size, num_classes):

    layers = []
    cin = input_size[0]
    cout = 16
    for i in range(cfg["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        layers.extend(conv_down(cout, 2 * cout))
        cin = 2 * cout
        cout = 2 * cout
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
    return nn.Sequential(conv_model, *out_layers)


# SOL@


class FancyCNN(nn.Module):
    """
    A fancy CNN model with :
        - stacked 3x3 convolutions
        - convolutive down sampling
        - a global average pooling at the end
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        layers = []
        cin = input_size[0]
        # @TEMPL
        # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # # TODO: Implement the model
        # self.model = nn.Sequential(*layers)
        # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # TEMPL@
        # @SOL
        cout = 16
        for i in range(cfg["num_layers"]):
            layers.extend(conv_relu_bn(cin, cout))
            layers.extend(conv_relu_bn(cout, cout))
            layers.extend(conv_down(cout, 2 * cout))
            cin = 2 * cout
            cout = 2 * cout
        conv_model = nn.Sequential(*layers)

        out_layers = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(cout, num_classes),
        ]
        self.model = nn.Sequential(conv_model, *out_layers)
        # SOL@

    def forward(self, x):
        # @TEMPL
        # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # # TODO: Implement the forward pass
        # return x
        # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # # TEMPL@
        # @SOL
        return self.model(x)
        # SOL@
