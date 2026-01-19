# coding: utf-8

# Standard imports
import logging
from typing import Tuple
import random
import sys
import functools
import operator

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
import numpy as np
import tqdm

# Local imports
from asrlab import utils

# Local imports
from asrlab import utils
from .dataset import DatasetFilter, load_dataset
from .charmap import CharMap
from .waveform import WaveformProcessor

def unpack_ravel(tensor: PackedSequence):
    unpacked_tensor, lens_tensor = pad_packed_sequence(tensor)  # T, B, *
    raveled = torch.cat(
        [tensori[:leni] for tensori, leni in zip(unpacked_tensor, lens_tensor)], 0
    )
    # raveled is (Tcum, num_features)
    return raveled

class BatchCollate(object):
    """
    Collator for the individual data to build up the minibatches
    """

    def __init__(
        self,
        sampling_rate: float,
        win_length: float,
        win_step: float,
        nmels: int,
        augment: bool,
        spectro_normalization: Tuple[float, float] = None,
    ):
        """
        Args:
            sampling_rate (float): the sampling rate (Hz)
            win_length (float): the window length (s.)
            win_step (float): the window step size (s.)
            nmels (int) : the number of mel scales to consider
            augment (bool) : whether to use data augmentation or not
            spectro_normalization (tuple(float, float)): mean, std
        """

        self.sampling_rate = sampling_rate
        self.waveform_processor = WaveformProcessor(
            sampling_rate,
            win_length,
            win_step,
            nmels,
            augment,
            spectro_normalization,
        )
        self.charmap = CharMap()

    def __call__(self, batch):
        """
        Builds and return a minibatch of data as a tuple (inputs, targets)
        All the elements are padded to be of equal time

        Returns:
            a tuple (spectros, targets) with :
                spectors : (Batch size, time, n_mels)
                targets : (Batch size, time)
        """
        # Extract the subcomponents
        # The CommonVoice dataset returns (waveform, sample_rate, dictionnary)
        # waveform is (1, seq_len)
        # dictionnary has the 'sentence' key for the transcript
        waveforms = [w.squeeze() for w, _, _ in batch]
        rates = [r for _, r, _ in batch]
        transcripts = [
            torch.LongTensor(self.charmap.encode(d["sentence"])) for _, _, d in batch
        ]

        # We resample the signal
        waveforms = [
            torchaudio.transforms.Resample(r, self.sampling_rate)(w)
            for w, r in zip(waveforms, rates)
        ]

        # Sort the waveforms and transcripts by decreasing waveforms length
        wt_sorted = sorted(
            zip(waveforms, transcripts), key=lambda wt: wt[0].shape[0], reverse=True
        )
        waveforms = [wt[0] for wt in wt_sorted]
        transcripts = [wt[1] for wt in wt_sorted]

        # Compute the lengths of the spectrograms from the lengths
        # of the waveforms
        waveforms_lengths = [w.shape[0] for w in waveforms]
        spectro_lengths = [
            self.waveform_processor.get_spectro_length(wl) for wl in waveforms_lengths
        ]
        transcripts_lengths = [t.shape[0] for t in transcripts]

        ###########################
        #### START CODING HERE ####
        ###########################

        ##
        # Step 1 : pad the sequence of tensors waveforms so that they all have the same
        #          length. The resulting tensor must be (T, B) where T is the maximal
        #          duration of the elements in waveforms.
        #          (1 line)
        # @TEMPL@waveforms = None
        waveforms = pad_sequence(waveforms)  # @SOL@

        # Step 2 : Apply the waveform_processor transform to the
        #          waveform tensor. Note the resulting tensor is (T, B, n_mels)
        #          (1 line)
        # @TEMPL@spectrograms = None
        spectrograms = self.waveform_processor(waveforms)  # @SOL@

        # Step 3 : pack the tensor of spectrograms given their length
        #          as computed in spectro_lengths
        #          (1 line)
        # @TEMPL@spectrograms = None
        # @SOL
        spectrograms = pack_padded_sequence(spectrograms, lengths=spectro_lengths)
        # SOL@

        # Step 3 : pad the sequence of tensors transcripts, so that all the rows
        #          of the resulting padded tensor have the same legnth. The resulting
        #          tensor is (Ty, B)
        #          (1 line)
        # @TEMPL@transcripts = None
        transcripts = pad_sequence(transcripts)  # @SOL@

        # Step 4 : pack the tensor of transcripts given their length as
        #          computed in transcripts_length
        #          Note : this packed tensor must be given enforce_sorted=False
        #          to ensure the i-th transcript stay aligned with the i-th
        #          spectrogram. Otherwise, torch would sort the transcripts
        #          independently from the spectrograms and the alignement between the
        #          spectrograms and the transcripts would be messed up
        #          (1 line)

        # @TEMPL@transcripts = None
        # @SOL
        transcripts = pack_padded_sequence(
            transcripts, lengths=transcripts_lengths, enforce_sorted=False
        )
        # SOL@

        ##########################
        #### STOP CODING HERE ####
        ##########################
        return spectrograms, transcripts


def get_dataloaders(
    cfg: dict,
    use_cuda: bool,
):
    """
    Build and return the pytorch dataloaders

    Args:
        cfg (dict): configuration of the dataloading 
        use_cuda : whether or not to use the pin_memory

    The keys in the cfg dictionnary are :
        root_dir (str or Path) : the root directory where the dataset
                                         is stored
        version (str) : the version of the dataset to consider, e.g. 1, 6.1, ..
        lang (str): the language of the dataset to load
        batch_size (int) : the number of samples per minibatch
        num_workers (int) : the number of threads to use for dataloading
        train_augment (bool) : whether to use SpecAugment
        nmels (int) : the number of mel scales to consider
        normalize : wheter or not to center reduce the spectrograms
        overwrite_index: whether or not to overwrite the cache files for the index of sequences to consider

        Optional:
        min_duration (float) : the minimal duration of the recordings to
                               consider
        max_duration (float) : the maximal duration of the recordings to
                               consider
        num_samples (int) : the number of samples to consider (usefull for small scale experiment)
    """
    commonvoice_root = cfg["root_dir"]
    commonvoice_version = cfg["version"]
    commonvoice_lang = cfg["lang"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    train_augment = cfg["train_augment"]
    nmels = cfg["nmels"]
    normalize = cfg["normalize"]
    mean_spectro = cfg.get("mean_spectro", None)
    std_spectro = cfg.get("std_spectro", None)

    overwrite_index = cfg["overwrite_index"]

    sampling_rate = cfg["sampling_rate"]
    win_length = cfg["win_length"]
    win_step = cfg["win_step"]

    min_duration = cfg.get("min_duration", None)
    max_duration = cfg.get("max_duration", None)
    num_samples = cfg.get("num_samples", None)

    def dataset_loader(fold, version, overwrite_index):
        ds = load_dataset(
            fold,
            root=commonvoice_root,
            version=commonvoice_version,
            lang=commonvoice_lang
        )
        if min_duration is None and max_duration is None:
            logging.info("Using the complete unfiltered dataset")
            return ds
        else:
            logging.info(f"Building a dataset filtered in duration in [{min_duration}, {max_duration}]")
            return DatasetFilter(
                ds=ds,
                min_duration=min_duration,
                max_duration=max_duration,
                cacheprefix=f"{fold}-{commonvoice_lang}-{version}",
                overwrite_index=overwrite_index,
            )

    train_dataset = dataset_loader("train", commonvoice_version, overwrite_index)
    valid_dataset = dataset_loader("dev", commonvoice_version, overwrite_index)
    test_dataset = dataset_loader("test", commonvoice_version, overwrite_index)
    
    # If requested, we take subsamples of our datasets
    if num_samples is not None:
        logging.info(f"Using a subset of samples of size {num_samples}")
        indices = range(num_samples)

        train_dataset = torch.utils.data.Subset(train_dataset, indices=indices)
        valid_dataset = torch.utils.data.Subset(valid_dataset, indices=indices)
        test_dataset = torch.utils.data.Subset(test_dataset, indices=indices)

    # If requested, we compute the normalization statistics
    if normalize:
        if mean_spectro is None or std_spectro is None:
            logging.info("Computing the normalization statistics")
            # Compute the normalization on the training set
            batch_collate_norm = BatchCollate(
                sampling_rate=sampling_rate,
                win_length=win_length,
                win_step=win_step, 
                nmels=nmels, 
                augment=False)
            norm_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=batch_collate_norm,
                pin_memory=use_cuda,
            )
            mean_spectro, mean2_spectro, std_spectro = 0.0, 0.0, 0.0
            N_elem = 0
            for spectros, _ in tqdm.tqdm(norm_loader):
                unpacked_raveled = unpack_ravel(spectros)
                mean_spectro += unpacked_raveled.sum().item()
                mean2_spectro += (unpacked_raveled**2).sum().item()
                N_elem += functools.reduce(operator.mul, unpacked_raveled.shape, 1)
            mean_spectro /= N_elem
            mean2_spectro /= N_elem
            std_spectro = np.sqrt(mean2_spectro - mean_spectro**2)
        
        else:
            logging.info("Using the provided mean_spectro/std_spectro")
        normalization = (mean_spectro, std_spectro)
        logging.info(f"Normalization coefficients : {mean_spectro}, {std_spectro}")
    else:
        normalization = None
        logging.info("Normalization coefficients computation skipped")

    # Build the collate functions. There are 2, one for training and one for inference (valid/test)
    batch_collate_train_fn = BatchCollate(
        sampling_rate=sampling_rate,
        win_length=win_length,
        win_step=win_step, 
        nmels=nmels, 
        augment=train_augment, 
        spectro_normalization=normalization
    )
    batch_collate_infer_fn = BatchCollate(
        sampling_rate=sampling_rate,
        win_length=win_length,
        win_step=win_step, 
        nmels=nmels, 
        augment=False, 
        spectro_normalization=normalization
    )

    # We can finally build the data loaders
    logging.info(f"Building a train loader with batch size = {batch_size}")
    logging.info(f"The dataset contains {len(train_dataset)} samples")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=batch_collate_train_fn,
        pin_memory=use_cuda,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batch_collate_infer_fn,
        pin_memory=use_cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=batch_collate_infer_fn,
        pin_memory=use_cuda,
    )

    return train_loader, valid_loader, test_loader, normalization

def ex_pack():

    batch_size = 10
    n_mels = 80
    max_length = 512

    # We create a collection of variable sizes sequences
    # Each element is of shape (Ti, n_mels) with Ti different
    # for each sequence
    tensors = [torch.randn(random.randint(1, max_length), n_mels) for i in range(batch_size)]

    # 1- To be packed, the tensors need to be sorted by
    #    decreasing length (see the doc of pack_padded_sequence)
    tensors = sorted(tensors,
                     key=lambda tensor: tensor.shape[0],
                     reverse=True)
    lengths = [t.shape[0] for t in tensors]

    # 2- We then pad the sequences to the length
    #    of the longest sequence
    tensors = pad_sequence(tensors)
    # tensors is (T, batch_size, n_mels)
    # note T is equal to the maximal length of the sequences

    # 3- Padded sequences can then be packed
    #    Note we need to provide the durations of the individual tensors
    packed_data = pack_padded_sequence(tensors, lengths=lengths)

    # Later, we can unpack the sequence
    # Note we recover the lengths that can be used to slice the "dense"
    # tensor unpacked_data appropriatly
    unpacked_data, lens_data = pad_packed_sequence(packed_data)

def test_dataloaders():
    utils.head("Testing the dataloaders")

    cfg = {
        "root_dir": "/mounts/datasets/datasets/CommonVoice",
        "version": "v24.0",
        "lang": "fr",
        "batch_size": 10,
        "num_workers": 2,
        "train_augment": False,
        "sampling_rate": 16000, # Hz,
        "win_length": 40*1e-3, # s.
        "win_step": 10*1e-3, # s.
        "nmels": 20,
        "normalize": True,
        "mean_spectro": -53.,
        "std_spectro": 29.,
        # "num_samples": 100,
        "overwrite_index": False,
        "min_duration": 1., # s.
        "max_duration": 9., # s.
    }

    # try:
    use_cuda = False
    loaders = get_dataloaders(
        cfg,
        use_cuda=use_cuda,
    )
    train_loader, valid_loader, test_loader, stats = loaders

    minibatch = next(iter(train_loader))

    utils.info(f"[1/] Got a minibatch of type {type(minibatch)}")
    if not isinstance(minibatch, tuple) or len(minibatch) != 2:
        utils.fail("Expected a minibatch to be a tuple spectrograms, transcripts")
    else:
        utils.info("The minibatch is a tuple with 2 elements")
        utils.succeed()

    packed_batch, packed_transcripts = minibatch

    utils.info(
        f"[2/] Got two items of type {type(packed_batch), type(packed_transcripts)}"
    )
    if not isinstance(packed_batch, PackedSequence) or not isinstance(
        packed_transcripts, PackedSequence
    ):
        utils.fail("Expected two PackedSequence")
    else:
        utils.info("The two items are correctly of type PackedSequence")
        utils.succeed()

    # except:
    #     utils.fail(f"{sys.exc_info()[0]}")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    # ex_pack()
    test_dataloaders()
