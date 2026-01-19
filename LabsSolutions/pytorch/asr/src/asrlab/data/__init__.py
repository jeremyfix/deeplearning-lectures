# coding: utf-8

# Standard imports

# External imports
import matplotlib.pyplot as plt
import torch

# Local imports
from .charmap import CharMap
from .waveform import WaveformProcessor
from .dataloader import get_dataloaders

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

