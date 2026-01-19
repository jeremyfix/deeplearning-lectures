# coding: utf-8

# Standard imports
import logging
import sys
import random

# External imports 
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from .charmap import CharMap
from .dataset import load_dataset
from .waveform import WaveformProcessor

def test_dataset():
    fold = "train"
    root = "/mounts/datasets/datasets/CommonVoice"
    version = "v24.0"
    lang = "fr"
    dataset = load_dataset(fold, root=root, version=version, lang=lang)

    resample_rate = 16000  # Hz
    win_length = 40 * 1e-3  # s.
    win_step = 10 * 1e-3 # s.
    num_mels = 80  #

    # Take one of the waveforms 
    idx = random.randint(0, len(dataset))
    logging.info(f"Loading the sample {idx}")
    waveform, sr, dictionary = dataset[idx]

    # Process the waveform
    waveform = waveform.transpose(0, 1)  # (B, T) to (T, B) as expected by WaveformProcessor
    trans_mel_spectro = WaveformProcessor(rate=resample_rate,
                                          win_length=win_length,
                                          win_step=win_step,
                                          nmels=num_mels,
                                          augment=False,
                                          spectro_normalization=None)
    mel_spectro = trans_mel_spectro(waveform).squeeze()  # (T, 1, N_MELS) to (T, N_MELS)

    # Process the sentence
    charmap = CharMap()
    sentence = dictionary["sentence"]
    encoded = charmap.encode(sentence)
    logging.info(f"Sentence : '{sentence}' encoded as '{encoded}'")

    logging.info(f"Number of timesteps in the spectrogram : {mel_spectro.shape[0]}")
    logging.info(f"Number of timesteps in the transcript : {len(encoded)}")

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(np.arange(waveform.squeeze().shape[0])/sr, waveform)
    ax.set_xlabel("Time (s.)")
    ax.set_ylabel("Amplitude")

    ax = fig.add_subplot(1, 2, 2)
    im = ax.imshow(mel_spectro.T,
                   extent=[0, mel_spectro.shape[0]*win_step,
                           0, mel_spectro.shape[1]],
                   aspect='auto',
                   cmap='magma',
                   origin='lower')
    ax.set_ylabel('Mel scale')
    ax.set_xlabel('TIme (s.)')
    ax.set_title('Log mel spectrogram')
    plt.colorbar(im)

    plt.suptitle(sentence)
    # plt.show()
    plt.savefig("dataset_sample.png")
    logging.info("Image saved as dataset_sample.png")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_dataset()
