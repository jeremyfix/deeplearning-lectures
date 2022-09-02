#!/usr/bin/env python3

# Standard imports
import sys
import inspect

# External imports
import torch
from torch.nn.utils.rnn import PackedSequence

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
    # Encoder with 1 input channel and 1 block
    encoder = models.UNetEncoder(1, 1)
    # succeed()
    fail(f"was expecting ")


if __name__ == "__main__":
    _RERAISE = True
    test_unet_encoder()
