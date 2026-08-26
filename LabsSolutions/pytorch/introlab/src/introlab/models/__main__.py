# coding: utf-8

# Standard imports
import logging

# External imports
import torch

# Local imports
from . import build_model


def useless_function():
    logging.info(
        "This is a useless function, just to show you how to invoke the functions defined in the models/__main__.py script"
    )


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    # @TEMPL
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # # TODO
    # # Fill in the expected output size
    # expected_output_size = None
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@
    expected_output_size = (batch_size, num_classes)  # @SOL@
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


# @SOL
def test_FFN():
    cfg = {"class": "FFN", "num_hidden": 32}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)
    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    expected_output_size = (batch_size, num_classes)  # @SOL@
    assert expected_output_size == output.shape
    print(f"Output tensor of size : {output.shape}")


def test_cnn():
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_cnn2():
    cfg = {"class": "FancyCNN", "num_layers": 4}
    input_size = (3, 128, 128)
    batch_size = 16
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


# SOL@

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    useless_function()
    test_linear()
    # @SOL
    test_FFN()
    test_cnn()
    test_cnn2()
    # SOL@
