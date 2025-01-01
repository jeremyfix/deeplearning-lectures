# coding: utf-8

# Standard import
import logging
import sys

# External imports
import torch

# Local imports
from . import build_model


def test_unet():
    logging.info("Testing UNet")
    cin = 1
    input_size = (cin, 256, 256)
    num_classes = 21
    X = torch.zeros((1, *input_size))
    model = build_model(
        {"class": "UNet", "encoder": {"model_name": "resnet18"}},
        input_size,
        num_classes,
    )
    model.eval()
    y = model(X)
    print(f"Output shape : {y.shape}")
    assert y.shape == (1, num_classes, input_size[1], input_size[2])


def test_deeplabv3():
    logging.info("Testing DeepLabV3Plus")
    cin = 1
    input_size = (cin, 256, 256)
    num_classes = 21
    X = torch.zeros((1, *input_size))
    model = build_model(
        {
            "class": "DeepLabV3Plus",
            "parameters": {"encoder_name": "resnet18", "encoder_weights": "imagenet"},
        },
        input_size,
        num_classes,
    )
    model.eval()
    y = model(X)
    print(f"Output shape : {y.shape}")
    assert y.shape == (1, num_classes, input_size[1], input_size[2])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # test_vision_maskrcnn()
    # test_mask_rcnn()
    test_unet()
    test_deeplabv3()
