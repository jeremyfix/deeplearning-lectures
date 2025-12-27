# coding: utf-8

# Local imports
from asrlab import utils

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
        nmels: int,
        augment: bool,
        spectro_normalization: Tuple[float, float] = None,
    ):
        """
        Args:
            nmels (int) : the number of mel scales to consider
            augment (bool) : whether to use data augmentation or not
            spectro_normalization (tuple(float, float)): mean, std
        """
        self.waveform_processor = WaveformProcessor(
            _DEFAULT_RATE,
            _DEFAULT_WIN_LENGTH * 1e-3,
            _DEFAULT_WIN_STEP * 1e-3,
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

        # We resample the signal to the _DEFAULT_RATE
        waveforms = [
            torchaudio.transforms.Resample(r, _DEFAULT_RATE)(w)
            if r != _DEFAULT_RATE
            else w
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
    commonvoice_root: str,
    commonvoice_version: str,
    cuda: bool,
    batch_size: int = 64,
    n_threads: int = 4,
    min_duration: float = 1,
    max_duration: float = 5,
    small_experiment: bool = False,
    train_augment: bool = False,
    nmels: int = _DEFAULT_NUM_MELS,
    logger=None,
    normalize=True,
    overwrite_index=False,
):
    """
    Build and return the pytorch dataloaders

    Args:
        commonvoice_root (str or Path) : the root directory where the dataset
                                         is stored
        commonvoice_version (str) : the version of the dataset to consider, e.g. 1, 6.1, ..
        cuda (bool) : whether to use cuda or not. Used for creating tensors
                      on the right device
        batch_size (int) : the number of samples per minibatch
        n_threads (int) : the number of threads to use for dataloading
        min_duration (float) : the minimal duration of the recordings to
                               consider
        max_duration (float) : the maximal duration of the recordings to
                               consider
        small_experiment (bool) : whether or not to use small subsets, usefull for debug
        train_augment (bool) : whether to use SpecAugment
        nmels (int) : the number of mel scales to consider
        logger : an optional logging logger
        normalize : wheter or not to center reduce the spectrograms
        overwrite_index: whether or not to overwrite the cache files for the index of sequences to consider
    """

    if small_experiment:
        min_duration = None
        max_duration = None

    def dataset_loader(fold, version, overwrite_index):
        ds = load_dataset(
            fold,
            commonvoice_root=commonvoice_root,
            commonvoice_version=commonvoice_version,
        )
        if not min_duration and not max_duration:
            return ds
        else:
            return DatasetFilter(
                ds=ds,
                min_duration=min_duration,
                max_duration=max_duration,
                cacheprefix=fold + "-" + version,
                overwrite_index=overwrite_index,
                logger=logger,
            )

    valid_dataset = dataset_loader("dev", commonvoice_version, overwrite_index)
    train_dataset = dataset_loader("train", commonvoice_version, overwrite_index)
    test_dataset = dataset_loader("test", commonvoice_version, overwrite_index)
    if small_experiment:
        indices = range(batch_size)

        train_dataset = torch.utils.data.Subset(train_dataset, indices=indices)
        valid_dataset = torch.utils.data.Subset(valid_dataset, indices=indices)
        test_dataset = torch.utils.data.Subset(test_dataset, indices=indices)

    if normalize:
        # Compute the normalization on the training set
        # batch_collate_norm = BatchCollate(nmels, augment=False)
        # norm_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=n_threads,
        #     collate_fn=batch_collate_norm,
        #     pin_memory=cuda,
        # )
        # mean_spectro, std_spectro = 0, 0
        # N_elem = 0
        # for spectros, _ in tqdm.tqdm(norm_loader):
        #     unpacked_raveled = unpack_ravel(spectros)
        #     mean_spectro += unpacked_raveled.sum().item()
        #     N_elem += functools.reduce(operator.mul, unpacked_raveled.shape, 1)
        # mean_spectro /= N_elem

        # for spectros, _ in tqdm.tqdm(norm_loader):
        #     unpacked_raveled = unpack_ravel(spectros)
        #     std_spectro += ((unpacked_raveled - mean_spectro) ** 2).sum()
        # std_spectro = (torch.sqrt(std_spectro / N_elem)).item()

        # Fix for speeding up debuggin
        mean_spectro = -31
        std_spectro = 32
        normalization = (mean_spectro, std_spectro)
    else:
        normalization = None

    if logger is not None:
        logger.info(f"Normalization coefficients : {mean_spectro}, {std_spectro}")

    batch_collate_train_fn = BatchCollate(
        nmels, augment=train_augment, spectro_normalization=normalization
    )
    batch_collate_infer_fn = BatchCollate(
        nmels, augment=False, spectro_normalization=normalization
    )

    print(f"Building a train loader with batch size = {batch_size}")
    print(f"The dataset contains {len(train_dataset)} samples")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
        collate_fn=batch_collate_train_fn,
        pin_memory=cuda,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        collate_fn=batch_collate_infer_fn,
        pin_memory=cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_threads,
        collate_fn=batch_collate_infer_fn,
        pin_memory=cuda,
    )

    return train_loader, valid_loader, test_loader

def test_dataloaders():
    utils.head("Testing the dataloaders")

    try:
        datasetroot = _DEFAULT_COMMONVOICE_ROOT
        datasetversion = _DEFAULT_COMMONVOICE_VERSION
        use_cuda = False
        B = 10
        nthreads = 2
        train_augment = False
        min_duration = 1  # s.
        max_duration = 5  # s.
        loaders = get_dataloaders(
            datasetroot,
            datasetversion,
            cuda=use_cuda,
            batch_size=B,
            n_threads=nthreads,
            min_duration=min_duration,
            max_duration=max_duration,
            small_experiment=False,
            train_augment=train_augment,
            nmels=_DEFAULT_NUM_MELS,
            logger=None,
        )
        train_loader, valid_loader, test_loader = loaders

        minibatch = next(iter(train_loader))

        utils.info(f"[1/] Got a minibatch of type {type(minibatch)}")
        if not isinstance(minibatch, tuple) or len(minibatch) != 2:
            utils.fail("Expected a minibatch to be a tuple spectrograms, transcripts")
        else:
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
            utils.succeed()

    except:
        utils.fail(f"{sys.exc_info()[0]}")

if __name__ == "__main__":
    test_dataloaders()
