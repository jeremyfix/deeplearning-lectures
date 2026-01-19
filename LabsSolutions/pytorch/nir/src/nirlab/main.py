# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torch.utils.tensorboard import SummaryWriter
import onnxruntime as ort
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import tqdm

# from torchmetrics.segmentation import MeanIoU
import deepcs.metrics
import deepcs.display
from deepcs.metrics import BatchAccuracy, BatchF1

# Local imports
from nirlab import data
from nirlab import models
from nirlab import optim
from nirlab import utils
from nirlab import metrics


def train(configpath):

    logging.info(f"Loading {configpath}")
    config = yaml.safe_load(open(configpath, "r"))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    (
        train_loader,
        valid_loader,
        dim_input,
        dim_output,
        normalizing_stats
    ) = data.get_dataloaders(data_config, use_cuda)

    # Request the first minibatch to get the dimensionalities of the input/output
    x0, y0 = next(iter(train_loader))
    dim_input = x0.shape[1]
    dim_output = y0.shape[1]

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(dim_input, dim_output, model_config)

    # Build the loss
    logging.info("= Loss")
    loss = torch.nn.MSELoss()

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Build the metrics
    train_fmetrics = {
        "mse": deepcs.metrics.GenericBatchMetric(loss),
    }
    test_fmetrics = {
        "mse": deepcs.metrics.GenericBatchMetric(loss),
    }

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Save the normalizing statistics
    with open(logdir / "normalizing_stats.yaml", "w") as file:
        yaml.dump(normalizing_stats, file)

    # Make a summary script of the experiment

    # x0 is a list of tensors
    input_size = (1,) + x0[0].shape
    logging.info(f"Input size : {input_size}")
    print(next(model.parameters()).device)

    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size, verbose=0)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n\n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    tensorboard_writer = SummaryWriter(log_dir=logdir)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, logdir, input_size, device, min_is_best=True
    )

    for e in range(config["nepochs"]):
        logging.info(f"\n\nEpoch {e}/{config['nepochs']} starting")

        # Train 1 epoch
        train_metrics = utils.train(
            model, train_loader, loss, optimizer, device, train_fmetrics
        )
        logging.info("Training epoch done")

        # Test
        valid_metrics = utils.test(
            model, valid_loader, device, test_fmetrics
        )
        logging.info("Validation done")

        checkpoint_metric_name = "mse"
        checkpoint_metric = valid_metrics[checkpoint_metric_name]

        updated = model_checkpoint.update(checkpoint_metric)

        # Display the metrics
        metrics_msg = "- Train : \n  "
        metrics_msg += "\n  ".join(
            f" {m_name}: {m_value}" for (m_name, m_value) in train_metrics.items()
        )
        metrics_msg += "\n"
        metrics_msg += "- Valid : \n  "
        metrics_msg += "\n  ".join(
            f" {m_name}: {m_value}"
            + ("[>> BETTER <<]" if updated and m_name == checkpoint_metric_name else "")
            for (m_name, m_value) in valid_metrics.items()
        )
        logging.info(metrics_msg)

        # Update the tensorboard
        for bname, bm in train_fmetrics.items():
            bm.tensorboard_write(tensorboard_writer, f"metrics/train_{bname}", e)
        for bname, bm in test_fmetrics.items():
            bm.tensorboard_write(tensorboard_writer, f"metrics/valid_{bname}", e)
        # Update the dashboard
        if wandb_log is not None:
            logging.info("Logging on wandb")

            data_to_log = {}
            for m_name, m_value in train_metrics.items():
                data_to_log[f"train_{m_name}"] = m_value
            for m_name, m_value in valid_metrics.items():
                data_to_log[f"valid_{m_name}"] = m_value

            wandb.log(data_to_log)

        logging.info(f" Epoch {e} done")


def test(logdir, img_path):
    logging.info(f"Loading model from {logdir}")

    logdir = pathlib.Path(logdir)

    providers = []
    use_cuda = True

    if use_cuda:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    inference_session = ort.InferenceSession(
        str(logdir / "best_model.onnx"), providers=providers
    )

    # Load our normalizing statistics
    stats = yaml.safe_load(open(str(logdir / "normalizing_stats.yaml"), "r"))
    mean = stats["mean"]
    std = stats["std"]

    raise NotImplemented("Not yet implemented")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        logging.error(f"Usage : {sys.argv[0]} <train|test> ...arguments...")
        sys.exit(-1)

    command = sys.argv[1]
    args = sys.argv[2:]

    # Before calling the command, we can sanity check the arguments
    if command == "train":
        if len(args) != 1:
            logging.error(f"Usage : {sys.argv[0]} train <config.yaml>")
            sys.exit(-1)
    elif command == "test":
        if len(args) != 2:
            logging.error(f"Usage : {sys.argv[0]} test logdir img_path")
            sys.exit(-1)
    else:
        logging.error(f"Unknown command {command}")
        sys.exit(-1)

    eval(f"{command}(*args)")
