# coding: utf-8

# Standard import
import logging
import sys

# External imports
import torch

# Local imports
from .unet import UNetEncoder, UNetDecoder, UNet
from .timm import GenericTimmEncoder
from . import build_model

def test_unet_encoder():
    logging.info("Testing UNet Encoder")
    # Encoder with 12 input channels
    batch_size, cin, height, width = 10, 12, 32, 64
    
    num_blocks = 4
    base_c  = 64
    encoder = UNetEncoder(cin, base_c=base_c, num_blocks=num_blocks)

    input_tensor = torch.zeros((batch_size, cin, height, width))
    output_tensor, encoder_features = encoder(input_tensor)
    output_shape = list(output_tensor.shape)

    expected_shape = [
        batch_size,
        2**num_blocks * base_c,
        height // 2**num_blocks,
        width // 2**num_blocks,
    ]
    if list(output_tensor.shape) == expected_shape:
        print(f"Congrats ! output shape is {expected_shape}")
    else:
        raise RuntimeError(f"was expecting {expected_shape} but got {output_shape}")

def test_unet_decoder():
    logging.info("Testing UNet Decoder")
    # Decoder
    batch_size, encoder_cout, height, width = 10, 512, 4, 8
    num_blocks, num_classes = 3, 14

    decoder = UNetDecoder(cin=encoder_cout, num_blocks=num_blocks, num_classes=num_classes)
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
        print(f"Congrats ! output shape is {expected_shape}")
    else:
        raise RuntimeError(f"was expecting {expected_shape} but got {output_shape}")

def test_unet():
    logging.info("Testing UNet")
    cin = 1
    input_size = (cin, 256, 256)
    num_classes = 21
    X = torch.zeros((1, *input_size))

    # @SOL 
    model = build_model(
        {"class": "UNet", "num_blocks": 4, "base_c": 18},
        input_size,
        num_classes,
    )
    # SOL@
    # @TEMPL
    # # vvvvvvvvv
    # # CODE HERE
    # model = None
    # # ^^^^^^^^^
    # TEMPL@

    model.eval()
    y = model(X)

    logging.info(f"Output shape : {y.shape}")
    assert y.shape == (1, num_classes, input_size[1], input_size[2])

# @SOL
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
# SOL@

def test_timm_encoder():
    logging.info("Testing the Timm encoder")
    cin = 3
    input_size = (cin, 256, 256)
    num_classes = 21
    cfg = {
        "class": "GenericUNet",
        "encoder": {
            "model_name": "resnet18",
            "pretrained": True
        }
    }
    model = GenericTimmEncoder(cin, 
                                cfg["encoder"]["model_name"], 
                                cfg["encoder"]["pretrained"])
    X = torch.zeros((1, *input_size))

    model.eval()
    y = model(X)
    
    logging.info(f"For an input of shape {X.shape}")
    logging.info("Output features of the encoder")
    for f in y:
        logging.info(f" - {f.shape}")

def test_generic_unet():
    logging.info("Testing the Generic UNet")
    cin = 1
    input_size = (cin, 256, 256)
    num_classes = 21
    X = torch.zeros((1, *input_size))
    model = build_model(
        {
            "class": "GenericUNet",
            "encoder": {
                "model_name": "resnet18",
                "pretrained": True
            }
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
    # @TEMPL
    # test_unet_encoder()
    # # test_unet_decoder()
    # # test_unet()
    # # test_timm_encoder()
    # # test_generic_unet()
    # TEMPL@
    # @SOL
    test_unet_encoder()
    test_unet_decoder()
    test_unet()
    test_deeplabv3()
    test_timm_encoder()
    test_generic_unet()
    # SOL@
