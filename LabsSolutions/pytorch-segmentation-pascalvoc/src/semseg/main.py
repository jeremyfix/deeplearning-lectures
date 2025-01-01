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
from semseg import data
from semseg import models
from semseg import optim
from semseg import utils
from semseg import metrics


def plot_results(normalizing_stats, imgs, preds, gts):
    mean, std = normalizing_stats["mean"], normalizing_stats["std"]

    fig, axes = plt.subplots(
        figsize=(6, 18), facecolor="w", nrows=imgs.shape[0], ncols=3
    )
    cmap = data.color_map()

    for i in range(imgs.shape[0]):
        input_i = imgs[i].permute(1, 2, 0).cpu().numpy()
        input_i = ((input_i * std) + mean).clip(0.0, 1.0)
        pred_i = preds[i].squeeze().cpu().numpy()
        gt_i = gts[i].squeeze().cpu().numpy()

        overlaid_pred = data.overlay(cmap, input_i, pred_i)
        overlaid_gt = data.overlay(cmap, input_i, gt_i)

        ax = axes[i, 0]
        ax.imshow(input_i)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = axes[i, 1]
        ax.imshow(overlaid_gt)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = axes[i, 2]
        ax.imshow(overlaid_pred)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


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
        input_size,
        num_classes,
        normalizing_stats,
        train_transforms,
    ) = data.get_dataloaders(data_config, use_cuda)

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.FocalLoss(ignore_index=255)

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
        "focal": deepcs.metrics.GenericBatchMetric(loss),
        "accuracy": BatchAccuracy(),
        # "confusion_matrix": metrics.BinaryConfusionMatrixMetric(),
    }
    test_fmetrics = {
        "focal": deepcs.metrics.GenericBatchMetric(loss),
        "accuracy": BatchAccuracy(),
        # "confusion_matrix": metrics.BinaryConfusionMatrixMetric(),
    }
    train_fmetrics["F1"] = BatchF1()
    test_fmetrics["F1"] = BatchF1()

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Save the normalizing statistics
    with open(logdir / "normalizing_stats.yaml", "w") as file:
        yaml.dump(normalizing_stats, file)

    # Make a summary script of the experiment
    x0, _ = next(iter(train_loader))
    # x0 is a list of tensors
    input_size = (1,) + x0[0].shape
    logging.info(f"Input size : {input_size}")
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
        model, logdir, input_size, device, min_is_best=False
    )

    for e in range(config["nepochs"]):
        logging.info(f"\n\nEpoch {e}/{config['nepochs']} starting")

        # Train 1 epoch
        train_metrics = utils.train(
            model, train_loader, loss, optimizer, device, train_fmetrics
        )
        logging.info("Training epoch done")
        train_macro_F1 = sum(train_metrics["F1"]) / num_classes
        train_metrics["macro_F1"] = train_macro_F1

        # Test
        valid_metrics = utils.test(model, valid_loader, device, test_fmetrics)
        logging.info("Validation done")

        # Compute the macro F1 for early stopping
        valid_macro_F1 = sum(valid_metrics["F1"]) / num_classes
        valid_metrics["macro_F1"] = valid_macro_F1

        checkpoint_metric_name = "macro_F1"
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

        # Perform some inferences on the validation set
        x, y = next(iter(valid_loader))
        x = x.to(device)
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)

        # Update the tensorboard
        for bname, bm in train_fmetrics.items():
            bm.tensorboard_write(tensorboard_writer, f"metrics/train_{bname}", e)
        for bname, bm in test_fmetrics.items():
            bm.tensorboard_write(tensorboard_writer, f"metrics/valid_{bname}", e)
        fig = plot_results(normalizing_stats, x, y_pred, y)
        tensorboard_writer.add_figure(
            "Sample predictions on the validation fold", fig, global_step=e
        )

        # Update the dashboard
        if wandb_log is not None:
            logging.info("Logging on wandb")

            data_to_log = {}
            for m_name, m_value in train_metrics.items():
                data_to_log[f"train_{m_name}"] = m_value
            for m_name, m_value in valid_metrics.items():
                data_to_log[f"valid_{m_name}"] = m_value

            # Log some images with their ground truth and predictions

            images = [
                wandb.Image(
                    x.cpu().numpy(),
                    masks={
                        "prediction": {"mask_data": y_pred.cpu().numpy()},
                        "ground_truth": {"mask_data": y.cpu().numpy()},
                    },
                )
                for x, y_pred, y in zip(x, y_pred, y)
            ]
            data_to_log["validation_sample"] = images
            wandb.log(data_to_log)

        logging.info(f" Epoch {e} done")


def test(logdir, img_path):
    logging.info(f"Loading model from {logdir}")

    logdir = pathlib.Path(logdir)

    providers = []
    use_cuda = True
    patch_size = 2000

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

    scan_img = np.array(Image.open(img_path))
    scan_img = scan_img[:8000, :8000]

    # Crop the image to only keep an integral number of patches
    # otherwise the inference will fail apparently because ORT will see
    # that the width/height of the patch has changed.
    # Maybe we need in that case to recreate the inference session object
    print("Cropping the image to keep a multiple of the patch size")
    scan_height = scan_img.shape[0]
    scan_width = scan_img.shape[1]
    scan_img = scan_img[
        : (scan_height // patch_size) * patch_size,
        : (scan_width // patch_size) * patch_size :,
    ]

    # Normalize our input
    scan_img = ((scan_img - mean * 255.0) / (std * 255.0)).astype(np.float32)
    scan_img = scan_img[np.newaxis, np.newaxis, ...]
    pred_probs = np.zeros_like(scan_img, dtype=np.float32)

    # Perform an inference over the patches
    for i in tqdm.tqdm(range(0, scan_img.shape[2], patch_size)):
        for j in range(0, scan_img.shape[3], patch_size):
            patch = scan_img[:, :, i : i + patch_size, j : j + patch_size]
            logits = inference_session.run(None, {"scan": patch})[0]
            probs = 1.0 / (1.0 + np.exp(-logits))
            pred_probs[:, :, i : i + patch_size, j : j + patch_size] = probs

    pred_mask = pred_probs >= 0.5

    # Plot the results
    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(scan_img.squeeze(), cmap="gray")
    plt.title("Zooscan image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_probs.squeeze(), interpolation="none", clim=(0.0, 1.0))
    plt.title("Probabilities")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), interpolation="none", cmap="tab20c")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(logdir / "prediction.png", bbox_inches="tight", dpi=600)
    print(f"Prediction saved in {logdir / 'prediction.png'}")


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
