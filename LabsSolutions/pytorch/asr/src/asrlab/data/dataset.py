# coding: utf-8

# Standard imports
import os
from pathlib import Path
from typing import Union, Tuple
import pickle
import sys
import time


# External imports
import torch.nn as nn
import torch.utils.data
from torchaudio.datasets import COMMONVOICE
from torchaudio.transforms import (
    AmplitudeToDB,
    MelSpectrogram,
    FrequencyMasking,
    TimeMasking,
)
import multiprocess

# Local imports
from asrlab import utils

def load_dataset(
    fold: str,
    commonvoice_root: Union[str, Path],
    commonvoice_version: str,
    lang: str = "fr",
) -> torch.utils.data.Dataset:
    """
    Load the commonvoice dataset within the path
    commonvoice_root/commonvoice_version/lang

    In this folder, we expect to find the tsv files of CommonVoice

    Args:
        fold (str): the fold to load, e.g. train, dev, test, validated, ..
        commonvoice_root

    Returns:
        torch.utils.data.Dataset: ``dataset``
    """
    datasetpath = os.path.join(commonvoice_root, commonvoice_version, lang)
    return COMMONVOICE(root=datasetpath, tsv=fold + ".tsv")


def is_sample_in_timerange(i, ds, min_duration, max_duration):
    (w, r, _) = ds[i]
    return i, (not min_duration or min_duration <= (w.squeeze().shape[0] / r)) and (
        not max_duration or (w.squeeze().shape[0] / r) <= max_duration
    )


class DatasetFilter(object):
    """
    Dataset object filtering an original dataset based on the
    durations of its waveform
    """

    def __init__(
        self,
        ds: torch.utils.data.Dataset,
        min_duration: float,
        max_duration: float,
        cacheprefix: str,
        overwrite_index: bool,
        logger=None,
    ) -> None:
        """
        Args:
            ds: the dataset to filter
            min_duration : the minimal duration in seconds
            max_duration : the maximal duration in seconds
            cacheprefix : the prefix of the cache file to which save the index files
        """

        def log(txt):
            if logger is not None:
                logger.info(txt)

        # At construction we build a list of indices
        # of valid samples from the original dataset
        cachepath = cacheprefix + f"-{min_duration}-{max_duration}.idx"
        if os.path.exists(cachepath) and not overwrite_index:
            log("Loading the pre-generated index file {cachepath}")
            self.valid_indices = pickle.load(open(cachepath, "rb"))
        else:
            log(
                f"Generating the index files, processing {len(ds)} files, saved in {cachepath}"
            )
            # self.valid_indices = [
            #     i
            #     for i, (w, r, _) in tqdm.tqdm(enumerate(ds))
            #     if (not min_duration or min_duration <= (w.squeeze().shape[0] / r))
            #     and (not max_duration or (w.squeeze().shape[0] / r) <= max_duration)
            # ]

            indices = list(range(len(ds)))
            t0 = time.time()
            with multiprocess.Pool(processes=None) as pool:
                results = list(
                    pool.map(
                        lambda idx, ds=ds, min_duration=min_duration, max_duration=max_duration: is_sample_in_timerange(
                            idx, ds, min_duration, max_duration
                        ),
                        indices,
                    ),
                )
            t1 = time.time()
            log(f"Elapsed : {t1-t0} seconds")
            self.valid_indices = [i for i, vi in results if vi]

            pickle.dump(self.valid_indices, open(cachepath, "wb"))
        self.ds = ds

    def __getitem__(self, idx):
        return self.ds[self.valid_indices[idx]]

    def __len__(self):
        return len(self.valid_indices)



class WaveformProcessor(object):
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
        """
        self.nfft = int(win_length * rate)
        self.nstep = int(win_step * rate)
        self.spectro_normalization = spectro_normalization

        ###########################
        #### START CODING HERE ####
        ###########################
        # @TEMPL@self.transform_tospectro = None
        # @SOL
        modules = [
            MelSpectrogram(
                sample_rate=rate, n_fft=self.nfft, hop_length=self.nstep, n_mels=nmels
            ),
            AmplitudeToDB(),
        ]
        self.transform_tospectro = nn.Sequential(*modules)
        # SOL@

        self.transform_augment = None
        if augment:
            time_mask_duration = 0.1  # s.
            time_mask_nsamples = int(time_mask_duration / win_step)
            nmel_mask = nmels // 4

            modules = [FrequencyMasking(nmel_mask), TimeMasking(time_mask_nsamples)]
            self.transform_augment = nn.Sequential(*modules)
        ##########################
        #### STOP CODING HERE ####
        ##########################

    def get_spectro_length(self, waveform_length: int):
        """
        Computes the length of the spectrogram given the length
        of the waveform

        Args:
            waveform_lengths: the number of samples of the waveform

        Returns:
            int: the number of time samples in the spectrogram
        """
        return waveform_length // self.nstep + 1

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

        # Apply data augmentation
        if self.transform_augment is not None:
            spectro = self.transform_augment(spectro)

        # spectrograms is (B, n_mel, T)
        # we permute it to be (T, B, n_mel)
        return spectro.permute(2, 0, 1)


def test_waveform_processor():
    utils.head("Testing the waveform processor")

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

        utils.info(f"[1/2] Got an output of shape {out.shape}")
        expected_shape = [Ts, B, _DEFAULT_NUM_MELS]
        if list(out.shape) == expected_shape:
            utils.succeed()
        else:
            utils.fail(f"was expecting {expected_shape}")

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
