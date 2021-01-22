#!/usr/bin/env python3
# coding: utf-8

# Standard imports
from typing import Optional, Tuple
from functools import reduce
import operator
# External imports
import torch
import torch.nn as nn


def conv_bn_leakyrelu(in_channels, out_channels):
    """
    Conv(3x3, same) - BN - LeakyRelu(0.2)
    """
    ks = 3
    return [
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=ks,
                  stride=1,
                  padding=int((ks-1)/2),
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.2)
    ]


def conv_downsampling(channels):
    """
    Conv(3x3, s2) - LeakyRelu(0.2)
    """
    ks = 3
    return [
        nn.Conv2d(channels, channels,
                  kernel_size=ks,
                  stride=2,
                  padding=int((ks-1)/2),
                  bias=True),
        nn.LeakyReLU(negative_slope=0.2)
    ]


class Discriminator(nn.Module):
    """
    The discriminator network tells if the input image is real or not
    The output logit is supposed to be high(-ly positive) for real images
    and low (highly negative) for fake images
    """

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 dropout: float,
                 base_c: int) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float) the probability of zeroing before the FC layer
            base_c (int): The base number of channels for the discriminator
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        in_C = img_shape[0]
        ######################
        # START CODING HERE ##
        ######################
        # Definition of the convolutional part of the classifier
        # Hint : conv_bn_leakyrelu() and conv_downsampling() can
        #        be usefull
        self.cnn = None
        ####################
        # END CODING HERE ##
        ####################

        # Compute the size of the representation by forward propagating
        # a fake tensor; This can be cpu tensor as the model is not yet
        # built and therefore not yet transfered to the GPU
        fake_input = torch.zeros((1, *img_shape))
        out_cnn = self.cnn(fake_input)
        print(f"The output shape of the convolutional part of the "
              f"discriminator is {out_cnn.shape}")
        num_features = reduce(operator.mul, out_cnn.shape[1:])

        ######################
        # START CODING HERE ##
        ######################
        # The fully connected part of the classifier
        self.classif = None
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
                    m.bias.fill_(0.)

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator
        Args:
            X(torch.Tensor (B, C, H, W)) : The images to classify

        Returns:
            Logits (torch.Tensor (B, )) : The logits
        """

        ######################
        # START CODING HERE ##
        ######################
        # Step 1 - Forward pass through the CNN part
        out_cnn = None

        # Step 2 - "Reshape" the 4D tensor to a 2D tensor
        #  Hint : Tensor.view can be of help
        input_classif = None

        # Step 3 - Forward pass through the fully connected layers
        out_classif = None
        ####################
        # END CODING HERE ##
        ####################
        return out_classif.squeeze()


def up_conv_bn_relu(in_channels, out_channels):
    """
    Upsampling with Upsample - Conv
    UpSample(x2) - Conv(3x3) - BN - Relu - Conv(3x3) - BN - Relu
    """
    ks = 3
    return [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=ks,
                  stride=1,
                  padding=int((ks-1)/2),
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels,
                  out_channels,
                  kernel_size=ks,
                  stride=1,
                  padding=int((ks-1)/2),
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]


def tconv_bn_relu(in_channels, out_channels, ksize, stride, pad, opad):
    """
    Upsampling with transposed convolutions
    TConv2D - BN - LeakyRelu(0.2)
    """
    return [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=ksize,
                               stride=stride,
                               padding=pad,
                               output_padding=opad),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
    ]


class Generator(nn.Module):
    """
    The generator network generates image from random inputs
    """

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 latent_size: int,
                 base_c: int) -> None:
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

        ######################
        # START CODING HERE ##
        ######################
        # Step 1 - Build the feedforward upscaling network
        self.upscale = nn.Sequential()

        # Step 2 - Build the convolutional upscaling network
        # Hint : up_conv_bn_relu() might be useful
        self.model = nn.Sequential()
        ####################
        # END CODING HERE ##
        ####################


        # Initialize the convolutional layers
        self.apply(self.init_weights)

    def init_weights(self, m):
        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.fill_(0.)

    def forward(self,
                X: Optional[torch.Tensor] = None,
                batch_size: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of the generator. You can either provide a noise
        input vector or specify the batch_size to let it generate the input

        Args:
            X (torch.Tensor, optional): The input noise batch
            batch_size (int, optional): The number of samples to generate
        """
        # X is expected to be a 2D tensor (B, L)
        if X is None:
            assert(batch_size is not None)
            device = next(self.parameters()).device
            X = torch.randn(batch_size, self.latent_size).to(device)
        else:
            if len(X.shape) != 2:
                raise RuntimeError("Expected a 2D tensor as input to the "
                                   f" generator got a {len(X.shape)}D tensor.")

        ######################
        # START CODING HERE ##
        ######################
        # Step 1 - Forward pass through the first linear layers
        #          to generate the seed image
        upscaled = None

        # Step 2 - "Reshape" the upscaled image as a 4D tensor
        #  Hint : use the view method
        reshaped = None

        # Step 3 : Forward pass through the last convolutional part
        #          to generate the image
        out = None
        ####################
        # END CODING HERE ##
        ####################

        return out


class GAN(nn.Module):

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 dropout: float,
                 discriminator_base_c: int,
                 latent_size: int,
                 generator_base_c: int) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float): The probability of zeroing before the FC layers
            discriminator_base_c (int) : The base number of channels for
                                         the discriminator
            latent_size (int) : The size of the latent space for the generator
            generator_base_c (int) : The base number of channels for the
                                     generator
        """
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.discriminator = Discriminator(img_shape,
                                           dropout,
                                           discriminator_base_c)
        self.generator = Generator(img_shape,
                                   latent_size,
                                   generator_base_c)

    def forward(self,
                X: Optional[torch.Tensor],
                batch_size: Optional[float]):
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
            real_logits = None
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
            fake_images = None
            fake_logits = None
            ####################
            # END CODING HERE ##
            ####################

            return fake_logits, fake_images



def test_discriminator():
    critic = Discriminator((1, 28, 28), 0.3, 32)
    X = torch.randn(64, 1, 28, 28)
    out = critic(X)
    assert(out.shape == torch.Size([64]))

def test_generator():
    generator = Generator((1, 28, 28), 100, 64)
    X = torch.randn(64, 100)
    out = generator(X, None)
    assert(out.shape == torch.Size([64, 1, 28, 28]))
    out = generator(None, 64)
    assert(out.shape == torch.Size([64, 1, 28, 28]))

if __name__ == '__main__':
    test_discriminator()
    test_generator()
