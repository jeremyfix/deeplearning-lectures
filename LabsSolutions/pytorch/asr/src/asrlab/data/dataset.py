# coding: utf-8

# Standard imports
import os
from pathlib import Path
from socket import SOL_ALG
from typing import Union
import pickle
import time
import logging

# External imports
import torch.nn as nn
import torch.utils.data
from torchaudio.datasets import COMMONVOICE
import multiprocess
import tqdm
import matplotlib.pyplot as plt

def load_dataset(
    fold: str,
    root: Union[str, Path],
    version: str,
    lang: str,
) -> torch.utils.data.Dataset:
    """
    Load the commonvoice dataset within the path
    commonvoice_root/commonvoice_version/lang

    In this folder, we expect to find the tsv files of CommonVoice

    Args:
        fold (str): the fold to load, e.g. train, dev, test, validated, ..
        root: the rootdir where to find the data
        version: the version of the dataset to consider
        lang: the language to load

    Returns:
        torch.utils.data.Dataset: ``dataset``
    """
    datasetpath = os.path.join(root, version, lang)
    if not os.path.exists(datasetpath):
        raise RuntimeError(f"The dataset path {datasetpath} does not exist.")

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
    ) -> None:
        """
        Args:
            ds: the dataset to filter
            min_duration : the minimal duration in seconds
            max_duration : the maximal duration in seconds
            cacheprefix : the prefix of the cache file to which save the index files
            overwrite_index: whether or not to overwirte the cache file
        """

        # At construction we build a list of indices
        # of valid samples from the original dataset
        cachepath = cacheprefix + f"-{min_duration}-{max_duration}.idx"
        if os.path.exists(cachepath) and not overwrite_index:
            logging.info(f"Loading the pre-generated index file {cachepath}")
            self.valid_indices = pickle.load(open(cachepath, "rb"))
        else:
            logging.info(f"Generating the index files, processing {len(ds)} files, saved in {cachepath}"
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
                        lambda idx, ds=ds, 
                            min_duration=min_duration, 
                            max_duration=max_duration: 
                            is_sample_in_timerange(idx, ds, min_duration, max_duration),
                        indices,
                    ),
                )
            t1 = time.time()
            logging.info(f"Elapsed : {t1-t0} seconds")
            self.valid_indices = [i for i, vi in results if vi]

            pickle.dump(self.valid_indices, open(cachepath, "wb"))
        self.ds = ds

    def __getitem__(self, idx):
        return self.ds[self.valid_indices[idx]]

    def __len__(self):
        return len(self.valid_indices)

# @SOL

def dataset_statistics():

    fold = "train"
    root = "/opt/datasets/CommonVoice"
    version = "v24.0"
    lang = "fr"

    ds = load_dataset(
        fold,
        root=root,
        version=version,
        lang=lang
    )

    def collate_durations(batch):
        """
        Tricky collate function just to extract the duration of the
        samples. This approach benefits from the multiprocessing of 
        the dataloaders

        Returns:
            a list of durations of the samples
        """
        return [len(xi.squeeze())/sri for xi, sri, _ in batch]

    durations = []

    # ds = torch.utils.data.Subset(ds, range(5000))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, collate_fn=collate_durations)
    for dur_i in tqdm.tqdm(dl):
        durations.extend(dur_i)
    print(durations)

    plt.figure()
    plt.ecdf(durations)
    plt.title(f"Duration histogram ({min(durations):.2f} s. <= d <= {max(durations):.2f} s.)")
    plt.xlabel("Duration (s.)")
    plt.ylabel("Probability duration < x")
    plt.savefig('durations_hist.png')

def check_votes():
    fold = "train"
    root = "/opt/datasets/CommonVoice"
    version = "v24.0"
    lang = "fr"

    ds = load_dataset(
        fold,
        root=root,
        version=version,
        lang=lang
    )
    for i in range(len(ds)):
        _, _, dicoi = ds[i]
        upi = dicoi["up_votes"]
        downi = dicoi["down_votes"]
        if downi > upi:
            print(f"{upi} {downi}")


# SOL@

def dataset_exploration():
    fold = "train"
    root = "/opt/datasets/CommonVoice"
    version = "v24.0"
    lang = "fr"

    ds = load_dataset(
        fold,
        root=root,
        version=version,
        lang=lang
    )
    
    # TODO: Explore the content of the dataset
    # Display the transcript of one of the samples
    # What is the duration, in seconds, of this sample ?
    
    # SOL@
    idx = 0

    line = ds._walker[idx]
    mp3path = os.path.join(ds._path, ds._folder_audio, line[1])
    print(mp3path)
    print(ds._folder_audio)
    xi, sri, dicoi = ds[idx]
    
    print(dicoi)

    duration = xi.squeeze().shape[0] / sri
    plt.figure()
    plt.plot(xi[0])
    plt.title(f'{dicoi["sentence"]}, {duration} s.')
    plt.savefig("sample.png")
    # SOL@


if __name__ == "__main__":
    dataset_exploration()
    # @SOL
    # check_votes()
    # dataset_statistics()
    # SOL@
