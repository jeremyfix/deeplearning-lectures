---
title:  'Automatic Speech Recognition'
author:
- Jeremy Fix
keywords: [PyTorch tutorial, Automatic Speech recognition, speech to text]
...

## Objectives

In this labwork, you will be experimenting with various recurrent neural networks for addressing the quite exciting problem of transcribing in text was is said in an audio file, a so called [Automatic Speech Recognition (ASR)](https://en.wikipedia.org/wiki/Speech_recognition) task.

The data we will be using are collected within the [Mozilla common voice](https://commonvoice.mozilla.org/) which are multi-language datasets of audio recordings and unaligned text transcripts. At the time of writing, there are almost 2000 hours in English and 700 hours in French. You can contribute yourself by either recording or validating recordings.

You will be experimenting with different models :

- Connectionist Temporal Classification (CTC) as introduced in [@Graves2014], which is the basis of the Deep Speech models (proposed by Baidu research[@Amodei2015] and also implemented by [Mozilla](https://github.com/mozilla/DeepSpeech)), 
- Seq2Seq with attention as introduced in the Listen Attend and Spell [@Chan2016]

Through this labwork, you will also learn about dealing with the specific tensor representations of variable length sequences.

## Setting up the dataloaders

In the CommonVoice dataset, you are provided with MP3 waveforms, usually sampled at 48 kHz (sometimes, slightly less on the version 6.1 corpus) with their unaligned transcripts. Unaligned means the annotation does not tell you when each word has been pronounced. No worries, the two models from the literature can deal with non aligned sequence to sequence.

The data are therefore : a waveform as input and a sequence of characters for the output. These two signals will be processed :

- instead of taking as input the waveform, we will be computing a spectrogram in Mel scale
- the characters will be filtered (to remove some variability) and converted to lists of integers

And these signals need to be stacked in mini-batch tensors. For doing so, you will have to fill in the [data.py](./data/02-pytorch-asr/data.py) script but before doing so, let us discuss some details. From a general point of view, it contains :

- load_dataset : the function for loading the CommonVoice datasets
- CharMap : the object in charge of performing the encoding/decoding from the characters to the integer representation
- WaveformProcessor : the object transforming the waveforms to the spectrograms
- BatchCollate : the object in charge of stacking multiple spectrograms and multiple encoded transcripts
- get_dataloaders : the function returning the train, valid and test dataloaders
- plot_spectro : a function for easily plotting a spectrogram and its transcript

The end result of this module is essentially the get_dataloaders which, by making use of the other functions/objects, build up the iterable minibatch collections. The CharMap will also be useful for converting the transcripts to and from their integer representations.

### Integer representations of the transcripts

The transcripts are encoded into integers and decoded from integers thanks to a vocabulary in the CharMap object, representend as the char2idx dictionnary and idx2char list. Let us play a little bit with this object.

**Exercice** Which function call gives you the vocabulary size ? What is this vocabulary size ? What is the encoding of the sentence "Je vais m'éclater avec des RNNS!". What do you obtain if you decode the encoded sentence ? Does it make sense to you ? The code you write for answering these questions is expected to be written within the `data.py` script in dedicated functions called within the *main* block :

```{.python}

if __name__ == '__main__':
    question1()
    question2()
    ...

```

<!--

vocab_size : 44
"[16, 27, 22, 1, 39, 18, 26, 36, 1, 30, 3, 22, 20, 29, 18, 37, 22, 35, 1, 18, 39, 22, 20, 1, 21, 22, 36, 1, 35, 31, 31, 36, 1, 5, 2]"
"¶je vais m'eclater avec des rnns .|"
-->


### Transforming the waveforms into spectrograms

Let us now have a look to the waveforms. Usually the models do not take the raw waveforms as input but preprocess them by computing a spectrogram using a short time fourier transform (STFT). The obtained spectrogram is then converted in logmel scale.

![The pipeline from the waveform to the logmel spectrogram](./data/02-pytorch-asr/waveform_to_spectro.png){width=100%}

**Exercice** In the WaveformProcessor, fill-in the code in the constructor for initializing the `transform` attribute. It must be a `torch.nn.Sequential` with the [MelSpectrogram](https://pytorch.org/audio/stable/transforms.html#melspectrogram) followed by a [conversion to DB](https://pytorch.org/audio/stable/transforms.html#amplitudetodb).

You can test your code with the following

```{.python}

import matplotlib.pyplot as plt

def test_spectro():
    dataset = load_dataset("train",
                           _DEFAULT_COMMONVOICE_ROOT,
                           _DEFAULT_COMMONVOICE_VERSION)

    # Take one of the waveforms 
    idx = 10
    waveform, rate, dictionary = dataset[idx]

    win_step = data._DEFAULT_WIN_STEP*1e-3
    trans_mel_spectro = WaveformProcessor(rate=rate,
                                          win_length=data._DEFAULT_WIN_LENGTH*1e-3,
                                          win_step=win_step,
                                          nmels=data._DEFAULT_NUM_MELS,
                                          augment=False)
    mel_spectro = trans_mel_spectro(waveform)[0]

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot()

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
    plt.tight_layout()
    plt.show()

```

### Collating variable size spectrograms and transcripts in minibatches

We know how to convert the transcripts into tensors of integers, we know how to convert waveforms to spectrograms, it remains to collate together multiple samples into minibatches but there are two difficulties :

- the waveforms have different time-spans and therefore their spectrograms have different time-spans
- the transcripts have different time-spans 

To represent sequences of variable length, pytorch provides you with the [PackedSequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html) datatype. We will not dig into the implementation details of this data structure. Sufficient to say that you are not expected to create PackedSequence directly but use the [pad_packed_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html) and [pack_padded_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) functions. Let us see these in actions on one example :

```{.python}

def ex_pack():
    import random
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

    batch_size = 10
    n_mels = 80
    max_length = 512
    # Given a list of batch_size tensors of variable sizes of shape (Ti, n_mels)
    tensors = [torch.randn(random.randint(1, max_length), n_mels)for i in range(batch_size)]

    # To be packed, the tensors need to be sorted by
    # decreasing length
    tensors = sorted(tensors,
                     key=lambda tensor: tensor.shape[0],
                     reverse=True)
    lengths = [t.shape[0] for t in tensors]

    # We start by padding the sequences to the max_length
    tensors = pad_sequence(tensors, batch_first=True)
    # tensors is (batch_size, T, n_mels)
    # note T is equal to the maximal length of the sequences

    # We can pack the sequence
    # Note we need to provide the timespans of the individual tensors
    packed_data = pack_padded_sequence(tensors, lengths=lengths,
                                      batch_first=True)

    # Later, we can unpack the sequence
    # Note we recover the lengths that can be used to slice the "dense"
    # tensor unpacked_data appropriatly
    unpacked_data, lens_data = pad_packed_sequence(packed_data,
                                                  batch_first=True)

```

Note that you may want to use usual "dense" tensors, padding at the end your tensors of variable size, but that is not a good idea because 1) the processing on the GPU can be limited to just the right amount if it knows the length of the sequences, 2) padding at the end is not a big deal with unidirectional RNNs but can possibly be a problem with bidirectional RNNs.

**Exercice** Equipped with these new PackedSequence tools, you now have to work on the build up of the minibatches.


Complete the *data.py* script and write a little piece for testing which should : 1) access a minibatch of one of your dataloader, 2) plot the spectrograms associated with their transcripts. For the plot part, you can take inspiration from the following python code. Note you are provided with the `plot_spectro` function.

Below is an example expected output :

![An example spectrogram in log-Mel scale of the validation set with its transcript](./data/02-pytorch-asr/spectro_valid.png){width=75%}



## Connectionist Temporal Classification (CTC)

The Connectionist Temporal Classification approach is based on a multi-layer recurrent neural network taking as input the successive frames of the spectrogram and outputting, at every time step, a probability distribution over the vocabulary to which we add a $\epsilon$ blank character. The CTC model is outputting one character at every iteration, therefore, it produces an output sequence of the same length of the input sequence, possibly with some $\epsilon$ blank characters and also some duplicated characters. 

![Illustration of the CTC model](./data/02-pytorch-asr/ctc.png){width=80%}

It also uses a specific loss which estimates the likelihood of the transcript, conditioned on the spectrogram, by summing over the probabilities of all the possible alignments with the blank character. Given a spectrogram $x$ and a transcript $y$ (e.g. ['t','u','_', 'e', 's']), its probability is computed as :

$$
p(y | x) = \sum_{a \in \mathcal{A}_{|x|}(y)} h_\theta(a | x) = \sum_{a_j \in \mathcal{A}_{|x|}(y)} \prod_t h_\theta(a_t | x)
$$

where $A_n(y)$ is the set of all the possible alignments of length $n$ of the sequence $y$. For example, for the sequence $y=(t,u,\_, e, s)$, some elements of $A_6(y)$ are $(\epsilon, \epsilon, t, u, \_, e, s), (\epsilon, t, \epsilon, u, \_,e,s), (t, \epsilon, u, \_,e,\epsilon, s), (t, t, \epsilon, u, \_, e, s), ...$ and this is tricky to compute efficiently. Fortunately, the CTC loss is [provided in pytorch](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html). The CTC loss of pytorch accepts tensors of probabilities of shape $(T_x, Batch, vocab\_size)$ and tensors of labels of shape $(batch, T_y)$ with $T_x$ respectively the maximal sequence length of the spectrogram and $T_y$ the maximal length of the transcript. An example call is given below : 

```{.python}

def ex_ctc():
    # The size of our minibatches
    batch_size = 32
    # The size of our vocabulary (including the blank character)
    vocab_size = 44
    # The class id for the blank token
    blank_id = 43

    max_spectro_length = 50
    min_transcript_length = 10
    max_transcript_length = 30

    # Compute a dummy vector of probabilities over the vocabulary (including the blank)
    # log_probs is here batch_first, i.e. (Batch, Tx, vocab_size)
    log_probs = torch.randn(batch_size, max_spectro_length, vocab_size).log_softmax(dim=1)
    spectro_lengths = torch.randint(low=max_transcript_length,
                                    high=max_spectro_length,
                                    size=(batch_size, ))

    # Compute some dummy transcripts
    # targets is here (batch_size, Ty)
    targets = torch.randint(low=0, high=vocab_size+1,  # include the blank character
                            size=(batch_size, max_transcript_length))
    target_lengths = torch.randint(low=min_transcript_length,
                                   high=max_transcript_length,
                                   size=(batch_size, ))
    loss = torch.nn.CTCLoss(blank=blank_id)

    # The log_probs must be given as (Tx, Batch, vocab_size)
    vloss = loss(log_probs.transpose(0, 1),
                 targets,
                 spectro_lengths,
                 target_lengths)

    print(f"Our dummy loss equals {vloss}")

```



## Data augmentation with SpecAugment

It was recently suggested, in the [SpecAugment](https://arxiv.org/abs/1904.08779) paper [@Park2019] that a valuable data augmentation is to partially mask the spectrograms both in the frequency and time domains.

Fortunately, torchaudio already implements the necessary functions for masking both in the time and frequency domains. I invite you to check the torchaudio documentation about the [FrequencyMasking](https://pytorch.org/audio/stable/transforms.html#frequencymasking) and [TimeMasking](https://pytorch.org/audio/stable/transforms.html#timemasking) transforms. 

These transforms can be implemented within the data.WaveformProcessor object. Just be sure to add a flag to the constructor to decide if you want or not to include these transforms. These transforms should probably be included only in the training set.

Below is an example of the resulting spectrogram, masked in time for up to $0.5$ s. and the mel scale for up to $27$ scales.

![An example log-Mel spectrogram with SpecAugment](./data/02-pytorch-asr/spectro_train.png){width=75%}


## Attention based recurrent neural network


## Going further

If you are interested in automatic speech recognition, you might be interested in the [End-to-End speech processing toolkit](https://github.com/espnet/espnet).

Also, to go further in terms of models, you might be interested in implementing beam search for decoding, eventually introducing some language model.

## References
