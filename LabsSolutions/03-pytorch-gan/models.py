#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import math
from typing import Optional, Tuple
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn


def conv_leakyrelu(in_channels, out_channels):
    """
    Conv(3x3, same) - BN - LeakyRelu(0.2)
    """
    ks = 3
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=False,
        ),
        nn.LeakyReLU(negative_slope=0.2),
    ]


def conv_leakyrelu_bn(in_channels, out_channels, ks=3):
    """
    Conv(3x3, same) - BN - LeakyRelu(0.2)
    """
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=False,
        ),
        nn.LeakyReLU(negative_slope=0.2),
        nn.BatchNorm2d(out_channels),
    ]


def conv_bn_leakyrelu(in_channels, out_channels):
    """
    Conv(3x3, same) - BN - LeakyRelu(0.2)
    """
    ks = 3
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.2),
    ]


def conv_downsampling(in_channels, out_channels):
    """
    Conv(3x3, s2) - LeakyRelu(0.2)
    """
    ks = 3
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=2,
            padding=int((ks - 1) / 2),
            bias=True,
        ),
        nn.LeakyReLU(negative_slope=0.2),
        nn.BatchNorm2d(out_channels),
    ]


class Discriminator(nn.Module):
    """
    The discriminator network tells if the input image is real or not
    The output logit is supposed to be high(-ly positive) for real images
    and low (highly negative) for fake images
    """

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        dropout: float,
        base_c: int,
        dnoise: float,
        num_classes: int,
    ) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float) the probability of zeroing before the FC layer
            base_c (int): The base number of channels for the discriminator
            dnoise (float): The amplitude of noise applied to the input of the discriminator
            num_classes (int): The number of output classes (usually 1, 2 or K)
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.dnoise = dnoise

        in_C = img_shape[0]
        ######################
        # START CODING HERE ##
        ######################
        # Definition of the convolutional part of the classifier
        # Hint : conv_bn_leakyrelu() and conv_downsampling() can
        #        be usefull
        # @TEMPL@self.cnn = None
        # @SOL
        # Note: the output receptive field size is 36 x 36
        #       the output representation size is 3 x 3
        self.cnn = nn.Sequential(
            *conv_leakyrelu_bn(in_C, base_c),
            *conv_leakyrelu_bn(base_c, base_c),
            *conv_downsampling(base_c, base_c),
            nn.Dropout2d(dropout),
            *conv_leakyrelu_bn(base_c, 2 * base_c),
            *conv_leakyrelu_bn(2 * base_c, 2 * base_c),
            *conv_downsampling(2 * base_c, 2 * base_c),
            nn.Dropout2d(dropout),
            *conv_leakyrelu_bn(2 * base_c, 2 * base_c),
            *conv_leakyrelu_bn(2 * base_c, 2 * base_c, 1),
            *conv_leakyrelu_bn(2 * base_c, 2 * base_c, 1),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        # SOL@
        ####################
        # END CODING HERE ##
        ####################

        # Compute the size of the representation by forward propagating
        # a fake tensor; This can be cpu tensor as the model is not yet
        # built and therefore not yet transfered to the GPU
        fake_input = torch.zeros((1, *img_shape))
        out_cnn = self.cnn(fake_input)
        print(
            f"The output shape of the convolutional part of the "
            f"discriminator is {out_cnn.shape}"
        )
        num_features = reduce(operator.mul, out_cnn.shape[1:])

        ######################
        # START CODING HERE ##
        ######################
        # The fully connected part of the classifier
        # @TEMPL@self.classif = None
        # @SOL
        self.classif = nn.Sequential(nn.Linear(num_features, num_classes))
        # SOL@
        ####################
        # END CODING HERE ##
        ####################

        # Run the initialization script
        self.apply(self.init_weights)

    def init_weights(self, m):
        """
        Initialize the weights of the convolutional layers
        """

        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.fill_(0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator
        Args:
            X(torch.Tensor (B, C, H, W)) : The images to classify

        Returns:
            Logits (torch.Tensor (B, )) : The logits
        """
        X = X + (self.dnoise * torch.randn(*X.shape, device=X.device))
        ######################
        # START CODING HERE ##
        ######################
        # Step 1 - Forward pass through the CNN part
        # @TEMPL@out_cnn = None
        out_cnn = self.cnn(X)  # @SOL@

        # Step 2 - "Reshape" the 4D tensor to a 2D tensor
        #  Hint : Tensor.view can be of help
        # @TEMPL@input_classif = None
        input_classif = out_cnn.view((out_cnn.shape[0], -1))  # @SOL@

        # Step 3 - Forward pass through the fully connected layers
        # @TEMPL@out_classif = None
        out_classif = self.classif(input_classif)  # @SOL@
        ####################
        # END CODING HERE ##
        ####################
        return out_classif.squeeze()


def conv_relu_bn(in_channels, out_channels):
    """
    Conv(3x3) - Relu - BN
    """
    ks = 3
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=False,
        ),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    ]


def up_conv_relu_bn(in_channels, out_channels):
    """
    Upsampling with Upsample - Conv
    UpSample(x2) - Conv(3x3) - BN - Relu - Conv(3x3) - BN - Relu
    """
    ks = 3
    return [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=False,
        ),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=False,
        ),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    ]


def tconv_bn_relu(in_channels, out_channels, ksize, stride, pad, opad):
    """
    Upsampling with transposed convolutions
    TConv2D - BN - LeakyRelu(0.2)
    """
    return [
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            output_padding=opad,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.2),
    ]


class Generator(nn.Module):
    """
    The generator network generates image from random inputs
    """

    def __init__(
        self, img_shape: Tuple[int, int, int], latent_size: int, base_c: int
    ) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            latent_size (int) : The dimension of the latent space
            base_c (int) : The base number of channels
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.base_c = base_c

        H, W = img_shape[1:]
        log2H = math.log2(H)
        log2W = math.log2(W)
        if int(log2H) != log2H or int(log2W) != log2W:
            raise ValueError("We are expecting modulo 2 heights/widths")
        if H != W:
            raise ValueError("We are expecting square images")

        ######################
        # START CODING HERE ##
        ######################
        # Step 1 - Build the feedforward upscaling network
        # @TEMPL@self.upscale = nn.Sequential()
        # @SOL

        self.first_c = self.base_c * (H // 16)
        print(f"[GENERATOR] The first convolutional block has {self.first_c} channels")

        self.upscale = nn.Sequential(
            nn.Linear(self.latent_size, 4 * 4 * self.first_c, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(4 * 4 * self.first_c),
        )
        # SOL@

        # Step 2 - Build the convolutional upscaling network
        # Hint : up_conv_bn_relu() might be useful
        # @TEMPL@self.model = nn.Sequential()
        # @SOL
        layers = []
        in_c = self.first_c
        for i in range(int(math.log2(H // 4))):
            layers.extend(up_conv_relu_bn(in_c, in_c // 2))
            in_c = in_c // 2

        print(
            f"[GENERATOR] The last upsample convolutional block outputs {in_c} channels"
        )

        layers.extend(
            [
                nn.Conv2d(
                    in_c,
                    self.img_shape[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)
        # SOL@
        ####################
        # END CODING HERE ##
        ####################

        # @SOL
        # Note : size, stride, pad, opad
        # self.model = nn.Sequential(
        #     *tconv_bn_relu2(base_c*4, base_c*2, 5, 1, 2, 0),
        #     # nn.Dropout2d(0.3),
        #     *tconv_bn_relu2(base_c*2, base_c, 5, 2, 2, 1),
        #     # nn.Dropout2d(0.3),
        #     nn.ConvTranspose2d(base_c, 1, 5, 2, 2, 1),
        #     nn.Tanh()  # as suggested by [Radford, 2016]
        # )
        # SOL@

        # Initialize the convolutional layers
        self.apply(self.init_weights)

    def init_weights(self, m):
        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.fill_(0.0)

    def forward(
        self, X: Optional[torch.Tensor] = None, batch_size: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass of the generator. You can either provide a noise
        input vector or specify the batch_size to let it generate the input

        Args:
            X (torch.Tensor, optional): The input noise batch
            batch_size (int, optional): The number of samples to generate
        """
        # X is expected to be a 2D tensor (B, L)
        if X is None:
            assert batch_size is not None
            device = next(self.parameters()).device
            X = torch.randn(batch_size, self.latent_size).to(device)
        else:
            if len(X.shape) != 2:
                raise RuntimeError(
                    "Expected a 2D tensor as input to the "
                    f" generator got a {len(X.shape)}D tensor."
                )

        ######################
        # START CODING HERE ##
        ######################
        # Step 1 - Forward pass through the first linear layers
        #          to generate the seed image
        # @TEMPL@upscaled = None
        upscaled = self.upscale(X)  # @SOL@

        # Step 2 - "Reshape" the upscaled image as a 4D tensor
        #  Hint : use the view method
        # @TEMPL@reshaped = None
        # @SOL
        reshaped = upscaled.view(-1, self.first_c, 4, 4)
        # SOL@

        # Step 3 : Forward pass through the last convolutional part
        #          to generate the image
        # @TEMPL@out = None
        out = self.model(reshaped)  # @SOL@
        ####################
        # END CODING HERE ##
        ####################

        return out


class GAN(nn.Module):
    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        dropout: float,
        discriminator_base_c: int,
        dnoise: float,
        num_classes: int,
        latent_size: int,
        generator_base_c: int,
    ) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float): The probability of zeroing before the FC layers
            discriminator_base_c (int) : The base number of channels for
                                         the discriminator
            dnoise (float): The amplitude of the normal noise applied as input
                            to the discriminator
            num_classes (int) : The number of class scores outputed by the discriminator
            latent_size (int) : The size of the latent space for the generator
            generator_base_c (int) : The base number of channels for the
                                     generator
        """
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.discriminator = Discriminator(
            img_shape, dropout, discriminator_base_c, dnoise, num_classes
        )
        self.generator = Generator(img_shape, latent_size, generator_base_c)

    def forward(self, X: Optional[torch.Tensor], batch_size: Optional[float]):
        """
        Given true images, returns the generated tensors
        and the logits of the discriminator for both the generated tensors
        and the true tensors

        Args:
            X (torch.Tensor) : a real image or None if we just
                               want the logits for the generated images
            batch_size (int) : the batch to consider when generating
                               fake images
        """

        if X is None and batch_size is None:
            raise RuntimeError("Not both X and batch_size can be None")
        if X is not None and batch_size is not None:
            raise RuntimeError("Not both X and batch_size can be not None")

        if X is not None:
            ######################
            # START CODING HERE ##
            ######################
            # An input tensor of real images is provided
            # we compute its logits
            # 1 line
            # @TEMPL@real_logits = None
            real_logits = self.discriminator(X)  # @SOL@
            ####################
            # END CODING HERE ##
            ####################
            return real_logits, X
        else:
            ######################
            # START CODING HERE ##
            ######################
            # No input tensor is provided. We generate batch_size fake images
            # and evaluate its logits
            # 2 lines
            # @TEMPL@fake_images = None
            # @TEMPL@fake_logits = None
            # @SOL
            fake_images = self.generator(X=None, batch_size=batch_size)
            fake_logits = self.discriminator(fake_images)
            # SOL@
            ####################
            # END CODING HERE ##
            ####################

            return fake_logits, fake_images


# @SOL
def test_tconv():
    layers = nn.Sequential(nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=2))
    print(layers)
    inputs = torch.zeros((1, 20, 2, 2))
    outputs = layers(inputs)
    print(outputs.shape)

    imagify = nn.Linear(100, 7 * 7 * 10)
    conv1 = nn.ConvTranspose2d(10, 10, kernel_size=5, stride=1, padding=2)
    conv2 = nn.ConvTranspose2d(
        10, 10, kernel_size=5, stride=2, padding=2, output_padding=1
    )
    conv3 = nn.ConvTranspose2d(
        10, 1, kernel_size=5, stride=2, padding=2, output_padding=1
    )

    X = torch.randn(64, 100)
    X = imagify(X).view(-1, 10, 7, 7)
    print("--")
    print(X.shape)
    X = conv1(X)
    print(X.shape)
    X = conv2(X)
    print(X.shape)
    X = conv3(X)
    print(X.shape)


# SOL@


def test_discriminator():
    critic = Discriminator((1, 32, 32), 0.3, 96, dnoise=0.1, num_classes=2)
    X = torch.randn(64, 1, 32, 32)
    out = critic(X)
    assert out.shape == torch.Size([64, 2])


def test_generator():
    generator = Generator((1, 32, 32), 100, 512)
    X = torch.randn(69, 100)
    out = generator(X, None)
    assert out.shape == torch.Size([69, 1, 32, 32])
    out = generator(None, 69)
    assert out.shape == torch.Size([69, 1, 32, 32])


if __name__ == "__main__":
    test_tconv()  # @SOL@
    test_discriminator()
    test_generator()
