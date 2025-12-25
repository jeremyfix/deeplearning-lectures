# coding: utf-8

# Standard imports
# External imports
import torch
import torch.nn as nn
import timm

def GenericTimmEncoder(cin, model_name, pretrained=True):
    # @TEMPL
    # # vvvvvvvvv
    # # CODE HERE
    # return None
    # # ^^^^^^^^^
    # TEMPL@
    # @SOL
    return timm.create_model(
        model_name=model_name,
        in_chans=cin,
        pretrained=pretrained,
        features_only=True,
    )
    # SOL@

def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
        nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]

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
    def __init__(self, encoder_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.decoder = nn.ModuleList()

        # Revert the list of channels since we expand from the
        # deepest layers of the encoder
        encoder_channels = encoder_channels[::-1]

        cin = encoder_channels[0]
        for cout in encoder_channels[1:]:
            self.decoder.append(
                GenericDecoderBlock(
                    cin=cin,
                    ccat=(cin // 2 + cout),
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

    # Forward propagation of a dummy tensor to get the encoder 
    # features dimensions
    X = torch.zeros((1, cin, 256, 256))
    encoder_features = encoder(X)
    encoder_channels = [fi.shape[1] for fi in encoder_features]

    decoder = GenericDecoder(encoder_channels, num_classes)
    return nn.Sequential(encoder, decoder)

