# coding: utf-8

# Standard imports
# External imports
import torch

# Local imports
from . import Discriminator, Generator
from ganlab import utils

def test_discriminator():
    utils.head("Testing the discriminator")
    critic = Discriminator(
        (1, 28, 28), dropout=0.3, base_c=32, dnoise=0.1, num_classes=2
    )
    B = 64

    X = torch.randn(B, 1, 28, 28)
    out = critic(X)

    utils.info(f"Got an output of shape {out.shape}")
    expected_shape = [B, 2]
    if list(out.shape) == expected_shape:
        utils.succeed()
    else:
        utils.fail(f" was expecting {expected_shape}")

def test_generator():
    utils.head("Testing the generator")

    generator = Generator((1, 32, 32), 100, 512)
    
    X = torch.randn(69, 100)
    out = generator(X, None)
    expected_shape = [69, 1, 32, 32]
    utils.succeed() if utils.test_equal(list(out.shape), expected_shape, 0) else fail(
        f" Got {out.shape}, was expecting {expected_shape}"
    )

    out = generator(None, 69)
    expected_shape = [69, 1, 32, 32]
    utils.succeed() if utils.test_equal(list(out.shape), expected_shape, 0) else fail(
        f" Got {out.shape}, was expecting {expected_shape}"
    )

if __name__ == "__main__":
    test_discriminator()
    test_generator()
