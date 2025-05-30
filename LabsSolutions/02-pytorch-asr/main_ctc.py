#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import os
import sys
import logging
import argparse
from pathlib import Path

# External imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchaudio
import deepcs.display
from deepcs.training import train as ftrain, ModelCheckpoint
from deepcs.testing import test as ftest
from deepcs.fileutils import generate_unique_logpath
import deepcs.metrics
import deepcs.rng
import wandb

# Local imports
import data
import models


def wrap_ctc_args(packed_predictions, packed_targets):
    """
    Returns:
        log_softmax predictions, targets, lens_predictions, lens_targets
    """
    unpacked_predictions, lens_predictions = pad_packed_sequence(
        packed_predictions
    )  # T, B, vocab_size

    # compute the log_softmax
    unpacked_predictions = unpacked_predictions.log_softmax(dim=2)  # T, B, vocab_size

    unpacked_targets, lens_targets = pad_packed_sequence(packed_targets)  # T, B
    unpacked_targets = unpacked_targets.transpose(0, 1)  # B, T
    # # Stack the subslices of the tensors
    # unpacked_targets = torch.cat(
    #     [batchi[:ti] for batchi, ti in zip(unpacked_targets, lens_targets)]
    # )

    return unpacked_predictions, unpacked_targets, lens_predictions, lens_targets


# def export_onnx(model, n_mels, device, filepath):
#     # The input shape is (T, B, mels)
#     # with T and B dynamic axes
#     export_input_size = (5, 1, n_mels)
#     dummy_input = torch.zeros(export_input_size, device=device)
#     # Important: ensure the model is in eval mode before exporting !
#     # the graph in train/test mode is not the same
#     # Although onnx.export handles export in inference mode now
#     model.eval()
#     torch.onnx.export(
#         model,
#         dummy_input,
#         filepath,
#         input_names=["input"],
#         output_names=["output"],
#         dynamic_axes={
#             "input": {0: "time", 1: "batch"},
#             "output": {0: "time", 1: "batch"},
#         },
#     )


def decode_samples(fdecode, loader, n, device, charmap):
    batch = next(iter(loader))
    spectro, transcripts = batch
    spectro = spectro.to(device)

    decoding_results = ""
    # unpacked_spectro is (T, B, n_mels)
    unpacked_spectro, lens_spectro = pad_packed_sequence(spectro)

    # unpacked_transcripts is (T, B)
    unpacked_transcripts, lens_transcripts = pad_packed_sequence(transcripts)

    # valid_batch is (T, B, n_mels)
    B = unpacked_spectro.shape[1]
    for idxv in range(min(n, B)):
        spectrogram = unpacked_spectro[:, idxv, :].unsqueeze(dim=1)
        spectrogram = pack_padded_sequence(spectrogram, lengths=[lens_spectro[idxv]])
        likely_sequences = fdecode(spectrogram)

        decoding_results += (
            "\nGround truth : " + charmap.decode(unpacked_transcripts[:, idxv]) + "\n"
        )
        decoding_results += "Log prob     Sequence\n"
        decoding_results += "\n".join(
            ["{:.2f}        {}".format(p, s) for (p, s) in likely_sequences]
        )
        decoding_results += "\n"

    return decoding_results


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    if args.wandb_project is not None and args.wandb_entity is not None:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, config=vars(args)
        )
        wandb_log = wandb.log
        logging.info("Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Data loading
    loaders = data.get_dataloaders(
        args.datasetroot,
        args.datasetversion,
        cuda=use_cuda,
        batch_size=args.batch_size,
        n_threads=args.nthreads,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        small_experiment=args.debug,
        train_augment=args.train_augment,
        nmels=args.nmels,
        logger=logger,
    )
    train_loader, valid_loader, test_loader = loaders

    # Parameters
    n_mels = args.nmels
    nhidden_rnn = args.nhidden_rnn
    nlayers_rnn = args.nlayers_rnn
    cell_type = args.cell_type
    dropout = args.dropout
    base_lr = args.base_lr
    num_epochs = args.num_epochs
    grad_clip = args.grad_clip

    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()
    blank_id = charmap.blankid

    # Model definition
    ###########################
    #### START CODING HERE ####
    ###########################
    # @TEMPL@model = None
    # @SOL
    # model = models.LinearModel(charmap, n_mels)
    model = models.CTCModel(
        charmap, n_mels, nhidden_rnn, nlayers_rnn, cell_type, dropout
    )

    # SOL@
    ##########################
    #### STOP CODING HERE ####
    ##########################

    decode = model.decode
    # decode = lambda spectro: model.beam_decode(
    #     spectro, beam_size=args.beamwidth, blank_id=blank_id
    # )

    model.to(device)

    # @SOL
    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from))
    # SOL@

    # Loss, optimizer
    baseloss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    loss = lambda *params: baseloss(*wrap_ctc_args(*params))

    ###########################
    #### START CODING HERE ####
    ###########################
    # @TEMPL@optimizer = None
    # @SOL
    if args.weight_decay is not None and args.weight_decay != 0.0:
        optimizer = optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
    # SOL@

    ##########################
    #### STOP CODING HERE ####
    ##########################

    metrics = {"CTC": deepcs.metrics.GenericBatchMetric(loss)}

    # Callbacks
    summary_text = (
        "## Summary of the model architecture\n"
        + f"{deepcs.display.torch_summarize(model)}\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
    )
    summary_text += "\n\n## Executed command :\n" + "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    if args.logname is not None:
        logdir = os.path.join(args.baselogdir, args.logname)
    else:
        logdir = generate_unique_logpath(args.baselogdir, "ctc")
    tensorboard_writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )
    if wandb_log is not None:
        wandb_log({"summary": summary_text})

    with open(os.path.join(logdir, "summary.txt"), "w") as f:
        f.write(summary_text)

    logger.info(f">>>>> Results saved in {logdir}")

    model_checkpoint = ModelCheckpoint(model, os.path.join(logdir, "best_model.pt"))
    if args.scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1
        # )
    else:
        scheduler = None

    # The location where to save the best model in ONNX
    # onnx_filepath = os.path.join(logdir, "best_model.onnx")

    logger.info(">>>>> Decodings before training")
    train_decodings = decode_samples(
        decode, train_loader, n=2, device=device, charmap=charmap
    )
    valid_decodings = decode_samples(
        decode, valid_loader, n=2, device=device, charmap=charmap
    )

    decoding_results = "## Decoding results on the training set\n"
    decoding_results += train_decodings
    decoding_results += "## Decoding results on the validation set\n"
    decoding_results += valid_decodings
    logger.info("\n" + decoding_results + "\n\n")

    # Training loop
    for e in range(num_epochs):
        logger.info("\n" + (">" * 20) + f" Epoch {e:05d}" + ("<" * 20) + "\n\n")

        train_metrics = ftrain(
            model,
            train_loader,
            loss,
            optimizer,
            device,
            metrics,
            grad_clip=grad_clip,
            num_model_args=1,
            num_epoch=e,
            tensorboard_writer=tensorboard_writer,
        )
        for m_name, m_value in train_metrics.items():
            tensorboard_writer.add_scalar(f"metrics/train_{m_name}", m_value, e + 1)

        # Compute and record the metrics on the validation set
        valid_metrics = ftest(model, valid_loader, device, metrics, num_model_args=1)
        better_model = model_checkpoint.update(valid_metrics["CTC"])

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "[%d/%d] Validation:   CTCLoss : %.3f %s"
            % (
                e,
                num_epochs,
                valid_metrics["CTC"],
                "[>> BETTER <<]" if better_model else "",
            )
        )

        for m_name, m_value in valid_metrics.items():
            tensorboard_writer.add_scalar(f"metrics/valid_{m_name}", m_value, e + 1)
        # Compute and record the metrics on the test set
        test_metrics = ftest(model, test_loader, device, metrics, num_model_args=1)
        logger.info(
            "[%d/%d] Test:   Loss : %.3f " % (e, num_epochs, test_metrics["CTC"])
        )
        for m_name, m_value in test_metrics.items():
            tensorboard_writer.add_scalar(f"metrics/test_{m_name}", m_value, e + 1)
        # Try to decode some of the validation samples
        model.eval()
        train_decodings = decode_samples(
            decode, train_loader, n=2, device=device, charmap=charmap
        )
        valid_decodings = decode_samples(
            decode, valid_loader, n=2, device=device, charmap=charmap
        )

        decoding_results = "## Decoding results on the training set\n"
        decoding_results += train_decodings
        decoding_results += "## Decoding results on the validation set\n"
        decoding_results += valid_decodings
        tensorboard_writer.add_text(
            "Decodings", deepcs.display.htmlize(decoding_results), global_step=e + 1
        )
        logger.info("\n" + decoding_results)

        # Log in wandb if available
        if wandb_log is not None:
            wandb_log(
                {"train_decodings": train_decodings, "valid_decodings": valid_decodings}
            )

            all_metrics = {}
            for m_name, m_value in train_metrics.items():
                all_metrics[f"train_{m_name}"] = m_value
            for m_name, m_value in valid_metrics.items():
                all_metrics[f"valid_{m_name}"] = m_value
            for m_name, m_value in test_metrics.items():
                all_metrics[f"test_{m_name}"] = m_value

            wandb_log(all_metrics)

        # if better_model:
        #     # Export the model in ONNX
        #     export_onnx(model, n_mels, device, onnx_filepath)


def test(args):
    """
    Test function to decode a sample with a pretrained model
    """
    import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)
    logger.info("Test")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()

    # Create the model
    # It is required to build up the same architecture than the one
    # used during training. If you do not remember the parameters
    # check the summary.txt file in the logdir where you have you
    # modelpath pt file saved. A better way to handle that
    # would be to use yaml files containing the hyperparameters for
    # training and load this yaml file when loading.
    n_mels = args.nmels
    nhidden_rnn = args.nhidden_rnn
    nlayers_rnn = args.nlayers_rnn
    cell_type = args.cell_type
    dropout = args.dropout

    modelpath = args.modelpath
    audiofile = args.audiofile
    beamwidth = args.beamwidth
    beamsearch = args.beamsearch
    assert modelpath is not None
    assert audiofile is not None

    logger.info("Building the model")
    model = models.CTCModel(
        charmap, n_mels, nhidden_rnn, nlayers_rnn, cell_type, dropout
    )
    model.to(device)
    model.load_state_dict(torch.load(modelpath))

    # Switch the model to eval mode
    model.eval()

    # Load and preprocess the audiofile
    logger.info("Loading and preprocessing the audio file")
    waveform, sample_rate = torchaudio.load(audiofile)
    waveform = torchaudio.transforms.Resample(sample_rate, data._DEFAULT_RATE)(
        waveform
    ).transpose(
        0, 1
    )  # (T, B)
    # Hardcoded normalization, this is dirty, I agree
    spectro_normalization = (-31, 32)
    # The processor for computing the spectrogram
    waveform_processor = data.WaveformProcessor(
        data._DEFAULT_RATE,
        data._DEFAULT_WIN_LENGTH * 1e-3,
        data._DEFAULT_WIN_STEP * 1e-3,
        n_mels,
        False,
        spectro_normalization,
    )
    spectrogram = waveform_processor(waveform).to(device)
    spectro_length = spectrogram.shape[0]

    # Plot the spectrogram
    logger.info("Plotting the spectrogram")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(
        spectrogram[0].cpu().numpy(), aspect="equal", cmap="magma", origin="lower"
    )
    ax.set_xlabel("Mel scale")
    ax.set_ylabel("Time (sample)")
    fig.tight_layout()
    plt.savefig("spectro_test.png")

    spectrogram = pack_padded_sequence(spectrogram, lengths=[spectro_length])

    logger.info("Decoding the spectrogram")

    if beamsearch:
        likely_sequences = model.beam_decode(spectrogram, beamwidth, charmap.blankid)
    else:
        likely_sequences = model.decode(spectrogram)

    print("Log prob    Sequence\n")
    print("\n".join(["{:.2f}      {}".format(p, s) for (p, s) in likely_sequences]))


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])
    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, help="The size of the minibatch", default=128
    )
    parser.add_argument(
        "--debug", action="store_true", help="Whether to use small datasets"
    )
    parser.add_argument(
        "--num_epochs", type=int, help="The number of epochs to train for", default=50
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        help="The base learning rate for the optimizer",
        default=0.0005,
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        help="The maxnorm of the gradient to clip to",
        default=None,
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Whether or not to use a learning rate scheduler.",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        help="The path to a tensor to use as initial conditions. Your model's parameters must be compatible with that tensor, otherwise, loading will fail.",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed used for the random number generators. To be used for reproducibitliy",
        default=None,
    )

    # Data parameters
    parser.add_argument(
        "--datasetversion",
        choices=["v1", "v6.1", "v12.0", "v15.0"],
        default=data._DEFAULT_COMMONVOICE_VERSION,
        help="Which CommonVoice corpus to consider",
    )
    parser.add_argument(
        "--datasetroot",
        type=str,
        default=data._DEFAULT_COMMONVOICE_ROOT,
        help="The root directory holding the datasets. "
        " These are supposed to be datasetroot/v1/fr or "
        " datasetroot/v6.1/fr",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use for " "loading the data",
        default=6,
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        help="The minimal duration of the waveform (s.)",
        default=1,
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        help="The maximal duration of the waveform (s.)",
        default=5,
    )
    parser.add_argument(
        "--nmels",
        type=int,
        help="The number of scales in the MelSpectrogram",
        default=data._DEFAULT_NUM_MELS,
    )

    # Model parameters
    parser.add_argument(
        "--nlayers_rnn", type=int, help="The number of RNN layers", default=4
    )
    parser.add_argument(
        "--nhidden_rnn",
        type=int,
        help="The number of units per recurrent layer",
        default=1024,
    )
    parser.add_argument(
        "--cell_type",
        choices=["GRU", "LSTM"],
        default="GRU",
        help="The type of reccurent memory cell",
    )

    # Regularization
    parser.add_argument(
        "--train_augment",
        action="store_true",
        help="Whether to use or not SpecAugment " "during training",
    )
    parser.add_argument(
        "--weight_decay", type=float, help="The weight decay coefficient", default=0.01
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="The dropout in the feedforward layers",
        default=0.1,
    )

    # For testing/decoding
    parser.add_argument("--modelpath", type=Path, help="The pt path to load")
    parser.add_argument(
        "--audiofile", type=Path, help="The path to the audio file to transcript"
    )
    parser.add_argument(
        "--beamwidth",
        type=int,
        help="The number of alternative decoding hypotheses" " to consider in parallel",
        default=10,
    )
    parser.add_argument(
        "--beamsearch",
        action="store_true",
        help="Whether or not to use beam search. If not, use" " max decoding.",
    )

    parser.add_argument(
        "--baselogdir",
        type=str,
        default="./logs",
        help="The base directory in which to save the logs",
    )

    parser.add_argument(
        "--logname",
        type=str,
        default=None,
        help="The name of the run used to define the logdir",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="The wandb project on which to record logs",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="The wandb entity on which to record logs",
    )
    args = parser.parse_args()

    if args.seed is not None:
        deepcs.rng.seed_torch(args.seed)

    eval(f"{args.command}(args)")
