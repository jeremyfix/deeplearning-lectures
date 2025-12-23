# coding: utf-8

# Standard imports
# External imports
import torch
import torch.nn as nn
import timm

def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
        nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]

class GenericTimmEncoder(nn.Module):
    def __init__(self, cin, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            in_chans=cin,
            pretrained=pretrained,
            features_only=True,
        )

    def forward(self, x):
        return self.model(x)


class GenericDecoderBlock(nn.Module):
    def __init__(self, cin, ccat):
        super().__init__()
        self.conv1 = nn.Sequential(*conv_relu_bn(cin, cin))
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2), *conv_relu_bn(cin, cin // 2)
        )
        self.conv2 = nn.Sequential(*conv_relu_bn(ccat, cin // 2))

    def forward(self, x, f_encoder):
        x = self.up_conv(self.conv1(x))
        x = torch.cat((x, f_encoder), dim=1)
        out = self.conv2(x)
        return out


class GenericDecoder(nn.Module):
    def __init__(self, encoder_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.decoder = nn.ModuleList()
        encoder_features = encoder_features[::-1]

        cin = encoder_features[0].shape[1]
        for i, f in enumerate(encoder_features[:-1]):
            self.decoder.append(
                GenericDecoderBlock(
                    cin=cin,
                    ccat=(cin // 2 + encoder_features[i + 1].shape[1]),
                )
            )
            cin = cin // 2
        self.out_conv = nn.Sequential(
            nn.Upsample(scale_factor=2), *conv_relu_bn(cin, num_classes)
        )

    def forward(self, encoder_features):
        encoder_features = encoder_features[::-1]
        x = encoder_features[0]
        for i, f in enumerate(encoder_features[1:]):
            x = self.decoder[i](x, f)
        y = self.out_conv(x)
        return y


def GenericUNet(cfg, input_size, num_classes):
    cin, _, _ = input_size
    encoder = GenericTimmEncoder(cin, **(cfg["encoder"]))

    # Forward propagation of a dummy tensor to get the encoder features dimensions
    X = torch.zeros((1, cin, 256, 256))
    encoder_features = encoder(X)

    decoder = GenericDecoder(encoder_features, num_classes)
    return nn.Sequential(encoder, decoder)

