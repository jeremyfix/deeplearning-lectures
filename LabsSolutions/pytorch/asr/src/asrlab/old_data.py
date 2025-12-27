# coding: utf-8

# Standard imports
import os
import functools
import logging
import operator
from pathlib import Path
from typing import Union, Tuple
import pickle
import sys
import time


# External imports
import torch.nn as nn
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    PackedSequence,
)
import torch.utils.data
import torchaudio
from torchaudio.datasets import COMMONVOICE
from torchaudio.transforms import (
    Spectrogram,
    AmplitudeToDB,
    MelSpectrogram,
    FrequencyMasking,
    TimeMasking,
)
import matplotlib.pyplot as plt
import multiprocess

# Local imports
from asrlab import utils


_DEFAULT_COMMONVOICE_ROOT = "/mounts/Datasets4/CommonVoice/"
_DEFAULT_COMMONVOICE_VERSION = "v15.0"
_DEFAULT_RATE = 16000  # Hz
_DEFAULT_WIN_LENGTH = 40  # ms
_DEFAULT_WIN_STEP = 10  # ms
_DEFAULT_NUM_MELS = 20  #




# @SOL

def plot_spectro(
    spectrogram: torch.Tensor,
    transcript: torch.Tensor,
    win_step: float,
    charmap: CharMap,
) -> None:
    """
    Args:
        spectrogram (time, n_mels) tensor
        trancript (target_len, ) LongTensor
        win_step is the stride of the windows, in seconds, for computing the
                 spectrogram
        charmap : object for converting between int and char for the transcripts
    """
    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot()

    im = ax.imshow(
        spectrogram.T,
        extent=[0, spectrogram.shape[0] * win_step, 0, spectrogram.shape[1]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Mel scale")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("{}".format(charmap.decode(transcript)))
    plt.colorbar(im)
    plt.tight_layout()


def ex_waveform_spectro():
    dataset = load_dataset(
        "train", _DEFAULT_COMMONVOICE_ROOT, _DEFAULT_COMMONVOICE_VERSION
    )

    # Take one of the waveforms
    idx = 10
    waveform, rate, dictionary = dataset[idx]
    walker = dataset._walker
    path_to_audio = os.path.join(dataset._path, dataset._folder_audio, walker[idx][1])
    print(f"I will be loading {path_to_audio}, with the transcript {walker[idx][2]}")

    n_begin = rate  # 1 s.
    n_end = 3 * rate  # 2 s.
    waveform = waveform[:, n_begin:n_end]  # B, T

    nfft = int(_DEFAULT_WIN_LENGTH * 1e-3 * _DEFAULT_RATE)
    # nmels = _DEFAULT_NUM_MELS
    nstep = int(_DEFAULT_WIN_STEP * 1e-3 * _DEFAULT_RATE)
    trans_spectro = nn.Sequential(
        Spectrogram(n_fft=nfft, hop_length=nstep), AmplitudeToDB()
    )
    spectro = trans_spectro(waveform)  # B, n_mels, T

    trans_mel_spectro = WaveformProcessor(
        rate=rate,
        win_length=_DEFAULT_WIN_LENGTH * 1e-3,
        win_step=_DEFAULT_WIN_STEP * 1e-3,
        nmels=_DEFAULT_NUM_MELS,
        augment=False,
        spectro_normalization=None,
    )
    mel_spectro = trans_mel_spectro(waveform.transpose(0, 1))  # T, B, n_mels
    plot_spectro(mel_spectro[:, 0, :], [], _DEFAULT_WIN_STEP * 1e-3, CharMap())

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))

    ax = axes[0]
    ax.plot([i / rate for i in range(n_begin, n_end)], waveform[0])
    ax.set_xlabel("Time (s.)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")

    ax = axes[1]
    im = ax.imshow(
        spectro[0],
        extent=[n_begin / rate, n_end / rate, 0, spectro.shape[1]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Frequency bins")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(
        mel_spectro[:, 0, :].T,
        extent=[n_begin / rate, n_end / rate, 0, mel_spectro.shape[0]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Mel scales")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("Mel-Spectrogram (dB)")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("waveform_to_spectro.png")
    logging.info("Image saved to waveform_to_spectro.png")
    plt.show()


def ex_spectro():

    charmap = CharMap()

    # Data loading
    batch_size = 4
    loaders = get_dataloaders(
        _DEFAULT_COMMONVOICE_ROOT,
        _DEFAULT_COMMONVOICE_VERSION,
        cuda=False,
        n_threads=4,
        min_duration=None,
        max_duration=None,
        batch_size=batch_size,
        train_augment=True,
        normalize=False,
    )
    train_loader, valid_loader, test_loader = loaders

    X, y = next(iter(train_loader))
    # X is (Tx, batch_size, n_mels)
    X, lens_X = pad_packed_sequence(X)
    # Y is (Ty, batch_size)
    y, lens_y = pad_packed_sequence(y)

    print("Some decoder texts from the LongTensors")
    for iy, li in enumerate(lens_y):
        print(charmap.decode(y[:li, iy]))

    fig, axes = plt.subplots(nrows=batch_size, ncols=1, sharex=True, figsize=(10, 7))
    for iax, ax in enumerate(axes):
        # spectroi is of shape (Tx, n_mels)
        print(X.shape)
        spectroi = X[:, iax, :]
        im = ax.imshow(
            spectroi.T,
            extent=[
                0,
                spectroi.shape[0] * _DEFAULT_WIN_STEP * 1e-3,
                0,
                spectroi.shape[1],
            ],
            aspect="auto",
            cmap="magma",
            origin="lower",
        )
        # vmin=-100, vmax=10)
        ax.set_ylabel("Mel scale")
        ax.set_title("{}".format(charmap.decode(y[: lens_y[iax], iax])))
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.xlabel("Time (s.)")
    plt.savefig("spectro.png")
    logging.info("Image saved to spectro.png")
    plt.show()


def ex_augmented_spectro():
    charmap = CharMap()

    # Data loading
    batch_size = 4
    loaders = get_dataloaders(
        _DEFAULT_COMMONVOICE_ROOT,
        _DEFAULT_COMMONVOICE_VERSION,
        cuda=False,
        n_threads=4,
        min_duration=None,
        max_duration=None,
        batch_size=batch_size,
        train_augment=True,
        normalize=False,
    )
    train_loader, valid_loader, test_loader = loaders

    # From the validation set
    X, y = next(iter(valid_loader))

    # X is (T, B, n_mels)
    X, lens_X = pad_packed_sequence(X)

    # Y is (T, B)
    y, lens_y = pad_packed_sequence(y)
    idx = 1
    plot_spectro(X[:, idx, :], y[: lens_y[idx], idx], _DEFAULT_WIN_STEP * 1e-3, charmap)
    print("spectro valid")
    plt.savefig("spectro_valid.png")

    # From the validation set
    X, y = next(iter(train_loader))

    # X is (T, B, n_mels)
    X, lens_X = pad_packed_sequence(X)
    # Y is (T, B)
    y, lens_y = pad_packed_sequence(y)
    idx = 0
    print(X.shape, _DEFAULT_WIN_STEP * 1e-3)
    plot_spectro(X[:, idx, :], y[: lens_y[idx], idx], _DEFAULT_WIN_STEP * 1e-3, charmap)
    print("spectro train")
    plt.savefig("spectro_train.png")
    logging.info("Image saved to spectro_train.png")

    plt.show()


def order_by_length():
    dataset_loader = functools.partial(
        load_dataset,
        commonvoice_root=_DEFAULT_COMMONVOICE_ROOT,
        commonvoice_version=_DEFAULT_COMMONVOICE_VERSION,
    )

    def forder(ds):
        idx_lens = [(w.shape[1], itrain) for itrain, (w, _, _) in enumerate(ds)]
        return sorted(idx_lens, key=lambda wi: wi[0])

    for k in ["dev", "test", "train"]:
        print(f"Ordering {k}")
        sorted_idx = forder(dataset_loader(k))
        with open(f"sorted_idx_{k}", "w") as f:
            f.write("\n".join(f"{idxi},{leni}" for leni, idxi in sorted_idx))


def test_spectro():
    dataset = load_dataset(
        "train", _DEFAULT_COMMONVOICE_ROOT, _DEFAULT_COMMONVOICE_VERSION
    )

    # Take one of the waveforms
    idx = 10
    waveform, rate, dictionary = dataset[idx]

    waveform = waveform.transpose(0, 1)
    print(waveform.shape)

    win_step = _DEFAULT_WIN_STEP * 1e-3
    trans_mel_spectro = WaveformProcessor(
        rate=rate,
        win_length=_DEFAULT_WIN_LENGTH * 1e-3,
        win_step=win_step,
        nmels=_DEFAULT_NUM_MELS,
        augment=False,
        spectro_normalization=None,
    )
    mel_spectro = trans_mel_spectro(waveform).squeeze()  # (T, N_MELS)

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot()

    im = ax.imshow(
        mel_spectro.T,
        extent=[0, mel_spectro.shape[0] * win_step, 0, mel_spectro.shape[1]],
        aspect="auto",
        cmap="magma",
        origin="lower",
    )
    ax.set_ylabel("Mel scale")
    ax.set_xlabel("TIme (s.)")
    ax.set_title("Log mel spectrogram")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("test_spectro.png")
    logging.info("Image saved to test_spectro.png")
    plt.show()


# SOL@
if __name__ == "__main__":
    # @TEMPL@pass
    test_waveform_processor()
    test_dataloaders()
    # @SOL
    # order_by_length()
    test_spectro()
    ex_waveform_spectro()
    ex_spectro()
    ex_augmented_spectro()
    # SOL@
