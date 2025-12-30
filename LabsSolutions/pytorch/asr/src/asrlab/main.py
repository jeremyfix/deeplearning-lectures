# coding: utf-8

# Standard imports
import os
import sys
import logging
import yaml
import pathlib

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
import deepcs.rng
import wandb

# Local imports
from . import data
from . import models
from . import utils
from . import metrics

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


def train(configpath):
    """
    Training of the algorithm
    """
    logging.info("Training")

    logging.info(f"Loading {configpath}")
    args = yaml.safe_load(open(configpath, "r"))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in args["logging"]:
        wandb_config = args["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Data loading
    logging.info("= Building the dataloaders")
    train_loader, valid_loader, test_loader, train_stats = data.get_dataloaders(
        args["data"],
        use_cuda=use_cuda,
    )

    # We need the char map to know about the vocabulary size
    charmap = data.CharMap()
    blank_id = charmap.blankid

    # Build the model
    logging.info("= Model")
    ###########################
    #### START CODING HERE ####
    ###########################
    modelconfig = args["model"]
    # @TEMPL@model = None
    # @SOL
    # model = models.LinearModel(charmap, n_mels)
    model = models.build_model(
        charmap, modelconfig
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
    if "resume_from" in args:
        model.load_state_dict(torch.load(args["resume_from"]))
    # SOL@

    # Build the loss
    logging.info("= Loss")
    baseloss = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)
    loss = lambda *params: baseloss(*wrap_ctc_args(*params))

    ###########################
    #### START CODING HERE ####
    ###########################
    # Build the optimizer
    logging.info("= Optimizer")
    optimconfig = args["optim"]
    # @TEMPL@optimizer = None
    # @SOL
    if "weight_decay" in optimconfig:
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=optimconfig["base_lr"], 
            weight_decay=optimconfig["weight_decay"]
        )
    else:
        optimizer = optim.Adam(model.parameters(), 
                               lr=optimconfig["base_lr"])
    # SOL@

    ##########################
    #### STOP CODING HERE ####
    ##########################

    # Build the callbacks
    logging_config = args["logging"]
    # Let us use as base logname the class name of the modek
    logname = modelconfig["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Build the metrics
    train_fmetrics = {"CTC": metrics.GenericBatchMetric(loss)}
    test_fmetrics = {"CTC": metrics.GenericBatchMetric(loss)}

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(args, file)

    # Save the normalizing statistics
    with open(logdir / "normalizing_stats.yaml", "w") as file:
        yaml.dump(train_stats, file)

    # Make a summary script of the experiment
    summary_text = (
        "## Summary of the model architecture\n"
        + f"{deepcs.display.torch_summarize(model)}\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
    )
    summary_text += "\n\n## Executed command :\n" + "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    tensorboard_writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )
    if wandb_log is not None:
        wandb_log({"summary": summary_text})

    # Define the early stopping callback
    x0, _ = next(iter(train_loader))
    # x0 is a list of tensors (spectro, transcript)
    input_size = (1,) + x0[0].shape
    model_checkpoint = utils.ModelCheckpoint(
        model, logdir, input_size, device, min_is_best=False
    )

    # Learning rate scheduler
    if "scheduler" in args:
        cfg = args["scheduler"]
        scheduler = lr_scheduler.StepLR(optimizer, 
                                        step_size=cfg["params"]["step_size"], 
                                        gamma=cfg["params"]["gamma"])
        # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1
        # )
    else:
        scheduler = None

    # The location where to save the best model in ONNX
    # onnx_filepath = os.path.join(logdir, "best_model.onnx")

    logging.info(">>>>> Decodings before training")
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
    logging.info("\n" + decoding_results + "\n\n")

    # Training loop
    nepochs = args["nepochs"]
    for e in range(nepochs):
        logging.info("\n" + (">" * 20) + f" Epoch {e:05d}" + ("<" * 20) + "\n\n")

        train_metrics = ftrain(
            model,
            train_loader,
            loss,
            optimizer,
            device,
            train_fmetrics,
            grad_clip=args["optim"]["grad_clip"],
            num_model_args=1,
            num_epoch=e,
            tensorboard_writer=tensorboard_writer,
        )
        for m_name, m_value in train_metrics.items():
            tensorboard_writer.add_scalar(f"metrics/train_{m_name}", m_value, e + 1)

        # Compute and record the metrics on the validation set
        valid_metrics = ftest(model, valid_loader, device, train_fmetrics, num_model_args=1)
        better_model = model_checkpoint.update(valid_metrics["CTC"])

        if scheduler is not None:
            scheduler.step()

        logging.info(
            "[%d/%d] Validation:   CTCLoss : %.3f %s"
            % (
                e,
                nepochs,
                valid_metrics["CTC"],
                "[>> BETTER <<]" if better_model else "",
            )
        )

        for m_name, m_value in valid_metrics.items():
            tensorboard_writer.add_scalar(f"metrics/valid_{m_name}", m_value, e + 1)

        # Compute and record the metrics on the test set
        test_metrics = ftest(model, test_loader, device, test_fmetrics, num_model_args=1)
        logging.info(
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
        logging.info("\n" + decoding_results)

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

    logging.info("Building the model")
    model = models.CTCModel(
        charmap, n_mels, nhidden_rnn, nlayers_rnn, cell_type, dropout
    )
    model.to(device)
    model.load_state_dict(torch.load(modelpath))

    # Switch the model to eval mode
    model.eval()

    # Load and preprocess the audiofile
    logging.info("Loading and preprocessing the audio file")
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
    logging.info("Plotting the spectrogram")
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

    logging.info("Decoding the spectrogram")

    if beamsearch:
        likely_sequences = model.beam_decode(spectrogram, beamwidth, charmap.blankid)
    else:
        likely_sequences = model.decode(spectrogram)

    print("Log prob    Sequence\n")
    print("\n".join(["{:.2f}      {}".format(p, s) for (p, s) in likely_sequences]))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        logging.error(f"Incorrect number of arguments. Usage is : ")
        logging.error(f" - {sys.argv[0]} train config.yaml")
        logging.error(f" - {sys.argv[0]} test")
        sys.exit(-1)

    if sys.argv[1] == "train":
        configpath = sys.argv[2]
        train(configpath)

    elif sys.argv[1] == "test": 
        raise NotImplementedError
        test(args)
        # For testing/decoding
        # parser.add_argument("--modelpath", type=Path, help="The pt path to load")
        # parser.add_argument(
        #     "--audiofile", type=Path, help="The path to the audio file to transcript"
        # )
        # parser.add_argument(
        #     "--beamwidth",
        #     type=int,
        #     help="The number of alternative decoding hypotheses" " to consider in parallel",
        #     default=10,
        # )
        # parser.add_argument(
        #     "--beamsearch",
        #     action="store_true",
        #     help="Whether or not to use beam search. If not, use" " max decoding.",
        # )
    else:
        raise RuntimeError(f"Unknown command {sys.argv[1]}")


