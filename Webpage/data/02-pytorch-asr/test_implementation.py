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
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def tab(n):
    return ' ' * 4*n

def fail(msg):
    print(colors.FAIL + tab(1) + f"[FAILED] From {inspect.stack()[1][3]}"
          + msg + colors.ENDC)

def succeed(msg = ""):
    print(colors.OKGREEN + tab(1) + "[PASSED]" + msg + colors.ENDC)

def head(msg):
    print(colors.HEADER + msg + colors.ENDC)

def info(msg):
    print(colors.OKBLUE + tab(1) + msg + colors.ENDC)

def test_equal(l1, l2, eps):
    return all([abs(l1i-l2i) <= eps for l1i, l2i in zip(l1,l2)])

def build_ctc_model(T, B):
    charmap = data.CharMap()

    return models.CTCModel(charmap,
                           n_mels=80,
                           num_hidden=185,
                           num_layers=3,
                           cell_type='GRU',
                           dropout=0.1)

def test_waveform_processor():
    head("Testing the waveform processor")

    try:
        wp = data.WaveformProcessor(rate=data._DEFAULT_RATE,
                                    win_length=data._DEFAULT_WIN_LENGTH*1e-3,
                                    win_step=data._DEFAULT_WIN_STEP*1e-3,
                                    nmels=data._DEFAULT_NUM_MELS,
                                    augment=False,
                                    spectro_normalization=None)

        torch.manual_seed(0)
        # Take some dummy waveforms
        T, B = 15000, 10
        waveforms = torch.randn((T, B))
        out = wp(waveforms)

        Ts = wp.get_spectro_length(T)

        info(f"[1/2] Got an output of shape {out.shape}")
        expected_shape = [Ts, B, data._DEFAULT_NUM_MELS]
        if list(out.shape) == expected_shape:
            succeed()
        else:
            fail(f"was expecting {expected_shape}")

        expected_out = [13.3708, 22.4177, -6.6840, 11.3768, 17.2921, 18.2367, 12.3000, 15.5621, 6.3457, 14.2817]
        info(f"[2/2] Got the output at [0, :, 0] = {out[0, :, 0].tolist()}")
        if test_equal(list(out[0, :, 0]), expected_out, eps=1e-2):
            succeed()
        else:
            fail(f"was expecting {expected_out}")
    except:
        fail(f"{sys.exc_info()[0]}")
        if _RERAISE: raise


def test_dataloaders():
    head("Testing the dataloaders")

    try:
        datasetroot = data._DEFAULT_COMMONVOICE_ROOT
        datasetversion = data._DEFAULT_COMMONVOICE_VERSION
        use_cuda = False
        B = 10
        nthreads = 2
        train_augment = False
        min_duration = 1  # s.
        max_duration = 5  # s.
        loaders = data.get_dataloaders(datasetroot,
                                       datasetversion,
                                       cuda=use_cuda,
                                       batch_size=B,
                                       n_threads=nthreads,
                                       min_duration=min_duration,
                                       max_duration=max_duration,
                                       small_experiment=False,
                                       train_augment=train_augment,
                                       nmels=data._DEFAULT_NUM_MELS,
                                       logger=None)
        train_loader, valid_loader, test_loader = loaders

        minibatch = next(iter(train_loader))

        info(f"[1/] Got a minibatch of type {type(minibatch)}")
        if not isinstance(minibatch, tuple) or len(minibatch) != 2:
            fail("Expected a minibatch to be a tuple spectrograms, transcripts")
        else:
            succeed()

        packed_batch, packed_transcripts = minibatch

        info(f"[2/] Got two items of type {type(packed_batch), type(packed_transcripts)}")
        if not isinstance(packed_batch, PackedSequence) or\
           not isinstance(packed_transcripts, PackedSequence):
               fail("Expected two PackedSequence")
        else:
            succeed()

    except:
        fail(f"{sys.exc_info()[0]}")
        if _RERAISE: raise


def test_model_cnn():
    head("Testing the cnn part")

    try:
        T, B = 124, 10
        model = build_ctc_model(T, B)

        cnn_inputs = torch.randn((T, B, model.n_mels)).transpose(0, 1).unsqueeze(dim=1)
        out_cnn = model.cnn(cnn_inputs)

        info(f"Got an output of shape {out_cnn.shape}")
        expected_shape = [10, 32, 31, 40]
        if list(out_cnn.shape) == expected_shape:
            succeed()
        else:
            fail(f"was expecting {expected_shape}")
    except:
        fail(f"{sys.exc_info()[0]}")
        if _RERAISE: raise



def test_model_rnn():
    head("Testing the rnn part")

    try:
        T, B = 124, 10
        model = build_ctc_model(T, B)

        rnn_inputs = torch.randn((T, B, 1280))
        out_rnn, _ = model.rnn(rnn_inputs)

        info(f"Got an output of shape {out_rnn.shape}")
        expected_shape = [124, 10, 370]
        if list(out_rnn.shape) == expected_shape:
            succeed()
        else:
            fail(f"was expecting {expected_shape}")
    except:
        fail(f"{sys.exc_info()[0]}")
        if _RERAISE: raise


def test_model_out():
    head("Testing the output part")

    try:
        T, B = 124, 10
        model = build_ctc_model(T, B)

        out_inputs = torch.randn((T, B, 370))
        out_out = model.charlin(out_inputs)

        info(f"Got an output of shape {out_out.shape}")
        expected_shape = [124, 10, 44]
        if list(out_out.shape) == expected_shape:
            succeed()
        else:
            fail(f"was expecting {expected_shape}")
    except:
        fail(f"{sys.exc_info()[0]}")
        if _RERAISE: raise


if __name__ == '__main__':
    _RERAISE = True

    test_waveform_processor()
    test_dataloaders()
    test_model_cnn()
    test_model_rnn()
    test_model_out()
