import torch
import torch.nn as nn
import torch.nn.init as init

import wide_resnet

def conv_bn_relu_maxp(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=False),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=False),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)]


class Linear(nn.Module):

    def __init__(self, input_dim, num_classes):
        super(Linear, self).__init__()

        self.classifier = nn.Linear(input_dim[0]*input_dim[1]*input_dim[2], num_classes)

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1)
        return self.classifier(inputs)


class CNN(nn.Module):

    def __init__(self, input_dim, num_classes, use_dropout):
        super(CNN, self).__init__()

        layers = conv_bn_relu_maxp(3, 64, 3)\
                +conv_bn_relu_maxp(64, 128, 3)\
                +conv_bn_relu_maxp(128, 256, 3)

        if use_dropout:
            self.features = nn.Sequential(*layers, nn.Dropout(0.5))
        else:
            self.features = nn.Sequential(*layers)

        probe_tensor = torch.zeros((1,) + input_dim)
        features = self.features(probe_tensor)
        print("Feature maps size : {}".format(features.shape))
        features_dim = features.view(-1)

        self.classifier = nn.Linear(features_dim.shape[0], num_classes)


    def forward(self, inputs):
        features = self.features(inputs)
        features = features.view(features.size()[0], -1)
        return self.classifier(features)


def conv(c_in, c_out, ks, stride):
    return [nn.Conv2d(c_in, c_out,
                      kernel_size=ks,
                      stride=stride,
                      padding=int((ks-1)/2), bias=True)]

def conv_relu_bn(c_in, c_out, ks, stride):
    return [nn.Conv2d(c_in, c_out,
                      kernel_size=ks,
                      stride=stride,
                      padding=int((ks-1)/2), bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c_out)]

def conv_bn_relu(c_in, c_out, ks, stride):
    return [nn.Conv2d(c_in, c_out,
                      kernel_size=ks,
                      stride=stride,
                      padding=int((ks-1)/2), bias=True),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)]



def bn_relu_conv(c_in, c_out, ks, stride):
    return [nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out,
                      kernel_size=ks,
                      stride=stride,
                      padding=int((ks-1)/2), bias=True)]



class WRN_Block(nn.Module):

    def __init__(self, c_in, c_out, dropout, stride):
        super(WRN_Block, self).__init__()

        layers = bn_relu_conv(c_in, c_out, 3)
        if dropout != 0.0:
            layers += [nn.Dropout(dropout)]
        layers += bn_relu_conv(c_out, c_out, 3)

        self.c33 = nn.Sequential(*layers)
        self.c11 = nn.Sequential(*conv(c_in, c_out, 1, stride=stride))


    def forward(self, inputs):
        return self.c33(inputs) + self.c11(inputs)

class WRN(nn.Module):

    def __init__(self, input_dim, num_classes, num_blocks, widen_factor, dropout, weight_decay):
        super(WRN, self).__init__()

        self.weight_decay = weight_decay

        num_filters = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        strides = [1, 2, 2]
        layers = [nn.Conv2d(input_dim[0], 16, kernel_size=3, stride=1, padding=1, bias=True)]
        for i_f, (nf_1, nf, stride) in enumerate(zip(num_filters[:-1], num_filters[1:], strides)):
            layers += [WRN_Block(nf_1  , nf, dropout, stride=stride)]
            layers += [WRN_Block(nf, nf, dropout, stride=1)] * (num_blocks-1)
        layers += [nn.AdaptiveAvgPool2d((1,1))]

        self.features = nn.Sequential(*layers)

        probe_tensor = torch.zeros((1,) + input_dim)
        features = self.features(probe_tensor)
        print("Feature maps size : {}".format(features.shape))
        features_dim = features.view(-1)

        self.classifier = nn.Linear(features_dim.shape[0], num_classes)
        self.init()

    def init(self):
        def finit(m):
            if type(m) == nn.Conv2d:
                init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                init.constant_(m.bias, 0)
            elif type(m) == nn.BatchNorm2d:
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            #elif type(m) == nn.Linear:
            #
        self.apply(finit)


    def penalty(self):
        penalty_term = None
        for m in self.modules():
            if type(m) in [nn.Conv2d, nn.Linear]:
                if not penalty_term:
                    penalty_term = m.weight.norm(2)**2
                else:
                    penalty_term += m.weight.norm(2)**2
        return self.weight_decay * penalty_term

    def forward(self, inputs):
        features = self.features(inputs)
        features = features.view(features.size()[0], -1)
        return self.classifier(features)


model_builder = {'linear': Linear,
                 'cnn': lambda idim, nc, dropout: CNN(idim, nc, dropout),
                 'wrn': lambda idim, nc, dropout, wd: WRN(idim, nc, 4, 10, dropout, wd),
                 'wide': lambda idim, nc, dropout, wd:wide_resnet.Wide_ResNet(28, 10, dropout, wd, nc) }


def build_model(model_name  : str,
                input_dim   : tuple,
                num_classes : int,
                dropout : float,
                weight_decay: float):
    return model_builder[model_name](input_dim, num_classes, dropout, weight_decay)


if __name__ == '__main__':

    input_dim   = (3, 32, 32)
    num_classes = 100
    model_name  = 'cnn'
    batch_size  = 4

    model = build_model(model_name, input_dim, num_classes)

    inputs = torch.randn((batch_size,) + input_dim)
    outputs = model(inputs)

    print(outputs.shape)
