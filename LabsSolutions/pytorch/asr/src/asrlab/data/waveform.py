# coding: utf-8

# Standard imports
from typing import Tuple
import sys

# External imports
import torch.nn as nn
import torch.utils.data
from torchaudio.transforms import (
    AmplitudeToDB,
    MelSpectrogram,
    FrequencyMasking,
    TimeMasking,
)

# Local imports
from asrlab import utils

class WaveformProcessor():
    def __init__(
        self,
        rate: float,
        win_length: float,
        win_step: float,
        nmels: int,
        augment: bool,
        spectro_normalization: Tuple[float, float],
    ):
        """
        Args:
            rate: the sampling rate of the waveform
            win_length: the length in second of the window for the STFT
            win_step: the length in second of the step size of the STFT window
            nmels:  the number of mel scales to consider
            augment (bool) : whether to use data augmentation or not
            spectro_normalization (float, float): Optional mean/std to normalize the spectrograms
        """
        self.spectro_normalization = spectro_normalization

        ###########################
        #### START CODING HERE ####
        ###########################
        # Convert the window length from seconds to number of samples
        self.n_fft = int(win_length * rate)
        # Convert the offset of the window from seconds to number of samples
        self.hop_length = int(win_step * rate) 
        self.n_mels = nmels

        # @TEMPL@self.transform_tospectro = nn.Sequential([])
        # @SOL
        modules = [
            MelSpectrogram(
                sample_rate=rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
            ),
            AmplitudeToDB(),
        ]
        self.transform_tospectro = nn.Sequential(*modules)
        # SOL@
        ##########################
        #### STOP CODING HERE ####
        ##########################
    
        # Adds some augmentations to the pipeline by either masking time or frequency
        # slices
        self.transform_augment = None
        if augment:
            time_mask_duration = 0.1  # s.
            time_mask_nsamples = int(time_mask_duration / win_step)
            nmel_mask = self.n_mels // 4

            modules = [FrequencyMasking(nmel_mask), TimeMasking(time_mask_nsamples)]
            self.transform_augment = nn.Sequential(*modules)

    def get_spectro_length(self, waveform_length: int):
        """
        Computes the length of the spectrogram given the length
        of the waveform

        Args:
            waveform_length: the number of samples of the waveform

        Returns:
            int: the number of time samples in the spectrogram
        """
        return waveform_length // self.hop_length + 1

    def __call__(self, waveforms: torch.Tensor):
        """
        Apply the transformation on the input waveform tensor
        The time dimension is smalled because of the hop_length given
        to the MelSpectrogram object.

        Args:
            waveforms(torch.Tensor) : (Tx, B) waveform
        Returns:
            spectrograms(torch.Tensor): (Tx//nstep + 1, B, n_mels)
        """
        # Compute the spectorgram
        waveforms = waveforms.transpose(0, 1)  # from (T, B) to (B, T)
        spectro = self.transform_tospectro(waveforms)  # (B, n_mels, T)

        # Normalize the spectrogram
        if self.spectro_normalization is not None:
            spectro = (
                spectro - self.spectro_normalization[0]
            ) / self.spectro_normalization[1]

        # Apply data augmentation if any
        if self.transform_augment is not None:
            spectro = self.transform_augment(spectro)

        # spectrograms is (B, n_mel, T)
        # we permute it to be (T, B, n_mel)
        return spectro.permute(2, 0, 1)


def test_waveform_processor():
    utils.head("Testing the waveform processor")

    _DEFAULT_RATE = 16000  # Hz
    _DEFAULT_WIN_LENGTH = 40  # ms
    _DEFAULT_WIN_STEP = 10  # ms
    _DEFAULT_NUM_MELS = 20  #

    try:
        wp = WaveformProcessor(
            rate=_DEFAULT_RATE,
            win_length=_DEFAULT_WIN_LENGTH * 1e-3,
            win_step=_DEFAULT_WIN_STEP * 1e-3,
            nmels=_DEFAULT_NUM_MELS,
            augment=False,
            spectro_normalization=None,
        )

        torch.manual_seed(0)
        # Take some dummy waveforms
        T, B = 15000, 10
        waveforms = torch.randn((T, B))
        out = wp(waveforms)

        Ts = wp.get_spectro_length(T)
    
        # Test for the dimensions of the spectrogram
        utils.info(f"[1/2] Got an output of shape {out.shape}")
        expected_shape = [Ts, B, _DEFAULT_NUM_MELS]
        if list(out.shape) == expected_shape:
            utils.succeed()
        else:
            utils.fail(f"was expecting {expected_shape}")

        # Test for the computed value of the spectrogram
        expected_out = [
            29.68541717529297,
            31.100982666015625,
            26.96457862854004,
            29.370576858520508,
            28.93488883972168,
            32.764102935791016,
            33.291133880615234,
            29.038545608520508,
            28.55718231201172,
            33.35734558105469,
        ]
        utils.info(f"[2/2] Got the output at [0, :, 0] = {out[0, :, 0].tolist()}")
        if utils.test_equal(list(out[0, :, 0]), expected_out, eps=1e-2):
            utils.succeed()
        else:
            utils.fail(f"was expecting {expected_out}")
    except:
        utils.fail(f"{sys.exc_info()[0]}")

if __name__ == "__main__":
    test_waveform_processor()
