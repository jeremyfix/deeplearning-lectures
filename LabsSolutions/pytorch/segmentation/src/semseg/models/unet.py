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

class UNetEncoderBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        # @SOL
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=cin,
                out_channels=cout,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=cout,
                out_channels=cout,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cout),
        )
        self.block3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # SOL@
        # @TEMPL
        # # vvvvvvvvv
        # # CODE HERE
        # self.block1 = None
        # self.block2 = None
        # self.block3 = None
        # # ^^^^^^^^^
        # TEMPL@

    def forward(self, inputs):
        # @SOL
        features = self.block2(self.block1(inputs))
        outputs = self.block3(features)
        # SOL@
        # @TEMPL
        # # vvvvvvvvv
        # # CODE HERE
        # features = None
        # outputs = None
        # # ^^^^^^^^^
        # TEMPL@
        return outputs, features


class UNetEncoder(nn.Module):
    def __init__(self, cin, base_c, num_blocks):
        super().__init__()
        # Note: use ModuleList to correctly register
        #       the modules it contains rather than plain list
        #  e.g. with plain list, the model.parameters() do not
        #       return the internal parameters of the modules contained
        #       in the list
        self.blocks = nn.ModuleList()

        self.cout = base_c
        for i in range(num_blocks):
            self.blocks.append(UNetEncoderBlock(cin, self.cout))
            # Prepare the parameters for the next layer
            cin = self.cout
            self.cout *= 2

        # Add the last encoding layer
        # which outputs 32 * 2*num_blocks channels
        # @SOL
        self.last_block = nn.Sequential(
            nn.Conv2d(cin, self.cout, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.cout),
        )
        # SOL@
        # @TEMPL
        # # vvvvvvvvv
        # # CODE HERE
        # self.last_block = None
        # # ^^^^^^^^^
        # TEMPL@

    def forward(self, inputs):
        # While iterating through the stages of the encoder
        # we keep a pointer to the outputs of "block2"
        # which will be latter used by the decoder
        prev_outputs, lfeatures = inputs, []
        for b in self.blocks:
            outb, featb = b(prev_outputs)
            # Keep track of the encoder features
            # to be given to the decoder pathway
            lfeatures.append(featb)
            # Prepare the input for the next block
            prev_outputs = outb
        outputs = self.last_block(prev_outputs)
        # Here :
        # outputs is the output tensor of the last encoding layer
        # lfeatures is the output features of the num_blocks blocks
        return outputs, lfeatures

class UNetDecoderBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        # @SOL
        self.conv1 = nn.Sequential(*conv_relu_bn(cin, cin))
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2), *conv_relu_bn(cin, cout)
        )
        self.conv2 = nn.Sequential(*conv_relu_bn(cin, cout))
        # SOL@
        # @TEMPL
        # # vvvvvvvvv
        # # CODE HERE
        # self.conv1 = None
        # self.up_conv = None
        # self.conv2 = None
        # # ^^^^^^^^^
        # TEMPL@

    def forward(self, x, f_encoder):
        # On passe à travers les premières couches convolutives et upsampling
        x = self.up_conv(self.conv1(x))  # @SOL@
        # # vvvvvvvvv
        # # CODE HERE
        # @TEMPL@x = None
        # # ^^^^^^^^^

        # On concatène les features de l'encoder
        # x et f_encoder sont (B, C, H, W)
        x = torch.cat((x, f_encoder), dim=1)  # @SOL@
        # # vvvvvvvvv
        # # CODE HERE
        # @TEMPL@x = None
        # # ^^^^^^^^^

        # On applique la dernière convolution
        out = self.conv2(x)  # @SOL@
        # # vvvvvvvvv
        # # CODE HERE
        # @TEMPL@out = None
        # # ^^^^^^^^^

        return out

class UNetDecoder(nn.Module):
    def __init__(self, cin, num_blocks, num_classes):
        super().__init__()
        # @SOL
        self.first_block = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(cin),
        )
        # SOL@
        # @TEMPL
        # # vvvvvvvvv
        # # CODE HERE
        # self.first_block = None
        # # ^^^^^^^^^
        # TEMPL@

        # Note: use ModuleList to correctly register
        #       the modules it contains rather than plain list
        #  e.g. with plain list, the model.parameters() do not
        #       return the internal parameters of the modules contained
        #       in the list
        self.blocks = nn.ModuleList()
        cout = cin // 2
        for i in range(num_blocks):
            self.blocks.append(UNetDecoderBlock(cin, cout))
            # Prepare the parameters for the next layer
            cin = cout
            cout = cout // 2

        # Add the last encoding layer
        # @SOL
        self.last_conv = nn.Conv2d(cin, num_classes, kernel_size=1, stride=1, padding=0)
        # SOL@
        # @TEMPL
        # # vvvvvvvvv
        # # CODE HERE
        # self.last_conv = None
        # # ^^^^^^^^^
        # TEMPL@

    def forward(self, encoder_outputs, encoder_features):
        outputs = self.first_block(encoder_outputs)
        for b, enc_features in zip(self.blocks, encoder_features[::-1]):
            outputs = b(outputs, enc_features)
        outputs = self.last_conv(outputs)
        return outputs

class UNet(nn.Module):
    """
    UNet model

    Args:
        cfg: configuration dictionary
        input_size: input image size (C, H, W)
        num_classes: number of output classes
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        cin, _, _ = input_size

        num_blocks = cfg["num_blocks"]
        base_c = cfg["base_c"]
        self.encoder = UNetEncoder(cin, base_c = base_c, num_blocks = num_blocks)
        encoder_cout = self.encoder.cout
        self.decoder = UNetDecoder(cin=encoder_cout, num_blocks=num_blocks, num_classes=num_classes)

    def forward(self, X):
        out, features = self.encoder(X)
        prediction = self.decoder(out, features)
        return prediction

