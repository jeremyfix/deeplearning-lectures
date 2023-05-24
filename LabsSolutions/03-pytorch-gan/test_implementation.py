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
    print(
        colors.FAIL
        + tab(1)
        + f"[FAILED] From {inspect.stack()[1][3]}"
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


def test_discriminator():
    head("Testing the discriminator")
    critic = models.Discriminator(
        (1, 28, 28), dropout=0.3, base_c=32, dnoise=0.1, num_classes=2
    )
    B = 64
    X = torch.randn(B, 1, 28, 28)
    out = critic(X)
    info(f"Got an output of shape {out.shape}")
    expected_shape = [B, 2]
    if list(out.shape) == expected_shape:
        succeed()
    else:
        fail(f" was expecting {expected_shape}")


if __name__ == "__main__":
    test_discriminator()
