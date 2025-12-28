# coding: utf-8

# Standard imports
import os
from pathlib import Path
from typing import Union
import pickle
import time
import logging

# External imports
import torch.nn as nn
import torch.utils.data
from torchaudio.datasets import COMMONVOICE
import multiprocess

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



