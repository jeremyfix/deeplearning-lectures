import torch
import torch.nn as nn


def get_model(config):
    '''
    Build the model specified by config['model']
    '''
    img_width = 28
    img_height = 28
    img_size = (1, img_height, img_width)
    num_classes = 10
    model = build_model(config['model'], img_size, num_classes)
    return model


def linear(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out)]


def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]


def dropout_linear(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out)]


def dropout_linear_relu(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]


def bn_dropout_linear(dim_in, dim_out, p_drop):
    return [nn.BatchNorm1d(dim_in),
            nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out)]


def bn_dropout_linear_relu(dim_in, dim_out, p_drop):
    return bn_dropout_linear(dim_in, dim_out, p_drop) + [nn.ReLU(inplace=True)]


def conv_relu_maxp(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)]


def conv_bn_relu(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]


class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class FullyConnected(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FullyConnected, self).__init__()
        self.classifier = nn.Sequential(
            *linear_relu(input_size, 256),
            *linear_relu(256, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class FullyConnectedRegularized(nn.Module):

    def __init__(self, input_size, num_classes, l2_reg):
        super(FullyConnectedRegularized, self).__init__()
        self.l2_reg = l2_reg
        self.lin1 = nn.Linear(input_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_classes)

        # self.classifier =  nn.Sequential(
        #    *linear_relu(input_size, 256),
        #    *dropout_linear_relu(256, 256, 0.2),
        #    *dropout_linear(256, num_classes, 0.2)
        #    )

    def penalty(self):
        return self.l2_reg * (self.lin1.weight.norm(2) +
                              self.lin2.weight.norm(2) +
                              self.lin3.weight.norm(2))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        y = self.lin3(x)
        return y


class VanillaCNN(nn.Module):

    def __init__(self, num_classes):
        super(VanillaCNN, self).__init__()

        # By default, Linear layers and Conv layers
        # use Kaiming He initialization
        self.features = nn.Sequential(
            *conv_relu_maxp(1, 16, 5),
            *conv_relu_maxp(16, 32, 5),
            *conv_relu_maxp(32, 64, 5)
        )
        # You must compute the number of features manualy to instantiate the
        # next FC layer
        # self.num_features = 64*3*3

        # Or you create a dummy tensor for probing the size of the feature maps
        probe_tensor = torch.zeros((1, 1, 28, 28))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
            *dropout_linear_relu(out_features.shape[0], 128, 0.5),
            *dropout_linear_relu(128, 256, 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  # OR  x = x.view(-1, self.num_features)
        y = self.classifier(x)
        return y


class FancyCNN(nn.Module):
    def __init__(self, num_classes):
        super(FancyCNN, self).__init__()

        base_n = 64
        self.features = nn.Sequential(
            *conv_bn_relu(1, base_n, 3), *conv_bn_relu(base_n, base_n, 3),
            nn.MaxPool2d(kernel_size=2),  # nn.Dropout(0.4),
            *conv_bn_relu(base_n, 2*base_n, 3),
            *conv_bn_relu(2*base_n, 2*base_n, 3),
            nn.MaxPool2d(kernel_size=2),  # nn.Dropout(0.4),
            *conv_bn_relu(2*base_n, 4*base_n, 3),
            *conv_bn_relu(4*base_n, 4*base_n, 3),
            nn.AvgPool2d(kernel_size=7)
        )

        self.lin1 = nn.Linear(4*base_n, num_classes)
        # self.classifier = nn.Sequential(
        #        nn.Dropout(0.5),
        #        nn.Linear(4*base_n, num_classes)
        # )

    # def penalty(self):
    #    return 1e-4 * self.lin1.weight.norm(2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        y = self.lin1(nn.functional.dropout(x, 0.5,
                                            self.training,
                                            inplace=True))
        # y = self.classifier(x)
        return y


def build_model(model_name, img_size, num_classes):
    model = None
    if(model_name == "linear"):
        model = LinearNet(img_size[0]*img_size[1]*img_size[2],
                          num_classes)
    elif(model_name == "fc"):
        model = FullyConnected(img_size[0]*img_size[1]*img_size[2],
                               num_classes)
    elif(model_name == "fcreg"):
        model = FullyConnectedRegularized(img_size[0]*img_size[1]*img_size[2],
                                          num_classes, 1e-3)
    elif(model_name == "vanilla"):
        model = VanillaCNN(num_classes)
    elif(model_name == "fancyCNN"):
        model = FancyCNN(num_classes)
    else:
        raise NotImplementedError("Unknown model {}".format(model_name))
    return model
