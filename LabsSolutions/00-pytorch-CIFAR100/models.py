import torch
import torch.nn as nn


def conv_bn_relu(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]


class Linear(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(Linear, self).__init__()

        self.classifier = nn.Linear(input_dim[0]*input_dim[1]*input_dim[2], num_classes)

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        return self.classifier(inputs)


class CNN(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(CNN, self).__init__()

        self.features = nn.Sequential(*conv_bn_relu(3, 16, 3))

        probe_tensor = torch.zeros((1,) + input_dim)
        features_dim = self.features(probe_tensor).view(-1)

        self.classifier = nn.Linear(features_dim.shape[0], num_classes)


    def forward(self, inputs):

        features = self.features(inputs)
        features = features.view(features.size()[0], -1)
        return self.classifier(features)

model_builder = {'linear': Linear,
                 'cnn': lambda idim, nc: CNN(idim, nc)}


def build_model(model_name  : str,
                input_dim   : tuple,
                num_classes : int):
    return model_builder[model_name](input_dim, num_classes)


if __name__ == '__main__':

    input_dim   = (3, 32, 32)
    num_classes = 100
    model_name  = 'cnn'
    batch_size  = 4

    model = build_model(model_name, input_dim, num_classes)

    inputs = torch.randn((batch_size,) + input_dim)
    outputs = model(inputs)

    print(outputs.shape)
