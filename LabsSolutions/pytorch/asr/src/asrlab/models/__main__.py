# coding: utf-8

# Standard imports
import sys

# External imports
import torch

# Local imports
from asrlab import data, models, utils

def test_model_cnn():
    utils.head("Testing the cnn part")
    
    cfg = {
        'class': "CTCModel",
        'params':
        {
            'n_mels': 80,
            'nhidden_rnn': 185,
            'nlayers_rnn': 3,
            'cell_type': "GRU",
            'dropout': 0.1
        }
    }

    T, B = 124, 10
    charmap = data.CharMap()
    model = models.build_model(charmap, cfg)

    cnn_inputs = torch.randn((T, B, model.n_mels)).transpose(0, 1).unsqueeze(dim=1)
    out_cnn = model.cnn(cnn_inputs)

    utils.info(f"Got an output of shape {out_cnn.shape}")
    expected_shape = [10, 32, 31, 40]
    if list(out_cnn.shape) == expected_shape:
        utils.succeed()
    else:
        utils.fail(f"was expecting {expected_shape}")


def test_model_rnn():
    utils.head("Testing the rnn part")

    cfg = {
        'class': "CTCModel",
        'params':
        {
            'n_mels': 80,
            'nhidden_rnn': 185,
            'nlayers_rnn': 3,
            'cell_type': "GRU",
            'dropout': 0.1
        }
    }

    T, B = 124, 10
    charmap = data.CharMap()
    model = models.build_model(charmap, cfg)

    rnn_inputs = torch.randn((T, B, 1280))
    out_rnn, _ = model.rnn(rnn_inputs)

    utils.info(f"Got an output of shape {out_rnn.shape}")
    expected_shape = [124, 10, 370]
    if list(out_rnn.shape) == expected_shape:
        utils.succeed()
    else:
        utils.fail(f"was expecting {expected_shape}")


def test_model_out():
    utils.head("Testing the output part")

    cfg = {
        'class': "CTCModel",
        'params':
        {
            'n_mels': 80,
            'nhidden_rnn': 185,
            'nlayers_rnn': 3,
            'cell_type': "GRU",
            'dropout': 0.1
        }
    }

    T, B = 124, 10
    charmap = data.CharMap()
    model = models.build_model(charmap, cfg)

    out_inputs = torch.randn((T, B, 370))
    out_out = model.charlin(out_inputs)

    utils.info(f"Got an output of shape {out_out.shape}")
    expected_shape = [124, 10, 44]
    if list(out_out.shape) == expected_shape:
        utils.succeed()
    else:
        utils.fail(f"was expecting {expected_shape}")


if __name__ == "__main__":
    test_model_cnn()
    test_model_rnn()
    test_model_out()
