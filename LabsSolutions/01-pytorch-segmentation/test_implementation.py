#!/usr/bin/env python3

# Standard imports
import sys
import inspect

# External imports
import torch

# Local imports
import data
import models

_RERAISE = False
_DEFAULT_T = 124
_DEFAULT_B = 10


class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def tab(n):
    return " " * 4 * n


def fail(msg):
    # print(inspect.stack())
    print(
        colors.FAIL
        + tab(1)
        + f"[FAILED] From {inspect.stack()[1][3]} "
        + msg
        + colors.ENDC
    )


def succeed(msg=""):
    print(colors.OKGREEN + tab(1) + "[PASSED]" + msg + colors.ENDC)


def head(msg):
    print(colors.HEADER + msg + colors.ENDC)


def info(msg):
    print(colors.OKBLUE + tab(1) + msg + colors.ENDC)


def test_equal(l1, l2, eps):
    return all([abs(l1i - l2i) <= eps for l1i, l2i in zip(l1, l2)])


def ftest(label):
    def test_call(func):
        def wrapper(*args, **kwargs):
            head(label)
            try:
                func(*args, **kwargs)
            except:
                fail(f"{sys.exc_info()[0]}")
                if _RERAISE:
                    raise

        return wrapper

    return test_call


@ftest("Testing Unet Encoder")
def test_unet_encoder():
    # Encoder with 12 input channels and 3 block
    batch_size, chan, height, width, num_blocks = 10, 12, 32, 64, 3
    encoder = models.UNetEncoder(chan, num_blocks)
    input_tensor = torch.zeros((batch_size, chan, height, width))
    output_tensor, encoder_features = encoder(input_tensor)
    output_shape = list(output_tensor.shape)
    expected_shape = [
        batch_size,
        2**num_blocks * 64,
        height // 2**num_blocks,
        width // 2**num_blocks,
    ]
    if list(output_tensor.shape) == expected_shape:
        succeed(f" output shape is {expected_shape}")
    else:
        fail(f"was expecting {expected_shape} but got {output_shape}")


@ftest("Testing UNet Decoder")
def test_unet_decoder():
    # Decoder
    batch_size, encoder_cout, num_blocks, num_classes = 10, 512, 3, 14
    height, width = 4, 8
    decoder = models.UNetDecoder(encoder_cout, num_blocks, num_classes)
    input_tensor = torch.zeros((batch_size, encoder_cout, height, width))
    encoder_features = [
        torch.zeros(
            (
                batch_size,
                64 * (2**i),
                height * (2 ** (num_blocks - i)),
                width * (2 ** (num_blocks - i)),
            )
        )
        for i in range(num_blocks)
    ]
    output_tensor = decoder(input_tensor, encoder_features)
    output_shape = list(output_tensor.shape)
    expected_shape = [
        batch_size,
        num_classes,
        height * 2**num_blocks,
        width * 2**num_blocks,
    ]
    if list(output_tensor.shape) == expected_shape:
        succeed(f" Output shape is {expected_shape}")
    else:
        fail(f"was expecting {expected_shape} but got {output_shape}")


if __name__ == "__main__":
    _RERAISE = True
    test_unet_encoder()
    test_unet_decoder()
