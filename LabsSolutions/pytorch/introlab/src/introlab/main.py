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

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # @SOL
    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None
    # SOL@

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes, classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    # @TEMPL
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # # TODO : Define the loss function
    # loss = None
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@
    loss = optim.get_loss(config["loss"])  # @SOL@

    # Build the optimizer
    logging.info("= Optimizer")
    # @TEMPL
    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # # TODO : Define the optimizer
    # optimizer = None
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # TEMPL@
    # @SOL
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())
    # SOL@

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    tensorboard_writer = SummaryWriter(logdir)

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        # @SOL
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        # SOL@
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    # @SOL
    if wandb_log is not None:
        wandb.log({"summary": summary_text})
    # SOL@

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss, train_acc = utils.train_one_epoch(model, train_loader, loss, optimizer, device)

        # Test
        test_loss, test_acc = utils.test(model, valid_loader, loss, device)

        updated = model_checkpoint.update(test_loss)
        is_better_msg = "[>> BETTER <<]" if updated else ""
        logging.info(f"[{e+1}/{config['nepochs']}] {is_better_msg} Test loss : {test_loss:.3f} ; Test acc : {test_acc:.2f}")

        # Update the dashboard
        metrics = {"train_CE": train_loss, "train_acc": train_acc, 
                   "test_CE": test_loss, "test_acc": test_acc}

        # On the tensorboard
        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, e)

        # @SOL
        # On Wandb if available
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)
        # SOL@


def test(config):
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} <train|test> config.yaml")
        sys.exit(-1)

    command = sys.argv[1]
    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[2], "r"))

    eval(f"{command}(config)")
