#!/usr/bin/env python3
# coding: utf-8
"""
    This script belongs to the lab work on semantic segmenation of the
    deep learning lectures https://github.com/jeremyfix/deeplearning-lectures
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# Standard modules
import logging
import argparse
import pathlib
import os
import sys

# External modules
import deepcs
import deepcs.training
import deepcs.testing
import deepcs.metrics
import deepcs.display
import deepcs.fileutils
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Local modules
import data
import models
import utils


def wrap_dtype(loss):
    def wrapped_loss(inputs, targets):
        return loss(inputs, targets.long())

    return wrapped_loss


def train(args):
    """Train a neural network on the stanford 2D-3D S semantic segmentation dataset

    Args:
        args (dict): parameters for the training

    Examples::

        python3 main.py train --normalize --model efficientnet_b3
    """
    logging.info("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Set up the train and valid transforms
    img_size = (256, 256)

    train_aug = A.Compose(
        [
            A.RandomCrop(768, 768),
            A.Resize(*img_size),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ]
    )
    valid_aug = A.Compose(
        [
            A.RandomCrop(768, 768),
            A.Resize(*img_size),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ]
    )

    def train_transforms(img, mask):
        aug = train_aug(image=np.array(img), mask=mask.numpy())
        return (aug["image"], aug["mask"])

    def valid_transforms(img, mask):
        aug = valid_aug(image=np.array(img), mask=mask.numpy())
        return (aug["image"], aug["mask"])

    train_loader, valid_loader, labels, unk_class_idx = data.get_dataloaders(
        args.datadir,
        use_cuda,
        args.batch_size,
        args.num_workers,
        args.debug,
        args.val_ratio,
        train_transforms,
        valid_transforms,
    )

    logging.info(f"Considering {len(labels)} classes : {labels}")

    # Make the model
    model = models.build_model(args.model, img_size, len(labels))
    model.to(device)

    # Make the loss
    # We ignore the pixels which are labeled as <UNK>
    ce_loss = wrap_dtype(nn.CrossEntropyLoss(ignore_index=unk_class_idx))

    # Make the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Metrics
    train_fmetrics = {
        "CE": deepcs.metrics.GenericBatchMetric(ce_loss),
        "F1": deepcs.metrics.BatchF1(),
    }
    test_fmetrics = {
        "CE": deepcs.metrics.GenericBatchMetric(ce_loss),
        "F1": deepcs.metrics.BatchF1(),
    }

    # Callbacks
    if args.logname is None:
        logdir = deepcs.fileutils.generate_unique_logpath(args.logdir, args.model)
    else:
        logdir = args.logdir / args.logname
    logdir = pathlib.Path(logdir)
    logging.info(f"Logging into {logdir}")

    if not logdir.exists():
        logdir.mkdir(parents=True)

    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + f"Commit id : {args.commit_id}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + " Arguments : {}".format(args)
        + "\n\n"
        + "## Summary of the model architecture\n"
        + f"{deepcs.display.torch_summarize(model, input_size)}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )

    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    # Callbacks
    tensorboard_writer = SummaryWriter(log_dir=logdir)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )

    model_checkpoint = deepcs.training.ModelCheckpoint(
        model, os.path.join(logdir, "best_model.pt"), min_is_best=False
    )

    valid_images, valid_gt = next(iter(valid_loader))
    valid_images = valid_images.to(device)

    for e in range(args.nepochs):
        deepcs.training.train(
            model,
            train_loader,
            ce_loss,
            optimizer,
            device,
            train_fmetrics,
            num_epoch=e,
            dynamic_display=True,
        )

        test_metrics = deepcs.testing.test(model, valid_loader, device, test_fmetrics)
        # Metrics recording on the tensorboard
        for bname, bm in test_fmetrics.items():
            bm.tensorboard_write(tensorboard_writer, f"metrics/test_{bname}", e)
        # Display the results on the console
        macro_test_F1 = sum(test_metrics["F1"]) / len(test_metrics["F1"])
        updated = model_checkpoint.update(macro_test_F1)
        metrics_msg = f"[{e}/{args.nepochs}] Test : \n  "
        metrics_msg += "\n  ".join(
            f" {m_name}: {m_value}" for (m_name, m_value) in test_metrics.items()
        )
        metrics_msg += f"\n  macro F1 : {macro_test_F1}" + (
            "[>> BETTER <<]" if updated else ""
        )
        logging.info(metrics_msg)

        # Get some test samples and predict the associated mask on them
        # Predict the labels
        with torch.no_grad():
            valid_predictions = model(valid_images).argmax(dim=1).detach().cpu().numpy()
        nsamples = 4
        fig, grid = plt.subplots(nrows=2, ncols=nsamples, sharex=True, sharey=True)
        grid = grid.T.ravel()
        # grid = ImageGrid(fig, 111, nrows_ncols=(2, 4), direction="column", axes_pad=0.1)
        for axgt, axp, vimg, vgt, vpred in zip(
            grid[::2], grid[1::2], valid_images, valid_gt, valid_predictions
        ):
            img_i = vimg.permute(1, 2, 0).cpu().numpy().squeeze()
            gt_i = vgt.numpy().squeeze()
            pred_i = vpred.squeeze()
            axgt.imshow(utils.overlay(img_i, gt_i))
            axp.imshow(utils.overlay(img_i, pred_i))
            # Remove the x-, y- ticks over the two axis
            axgt.tick_params(
                labelcolor="none",
                which="both",
                top=False,
                bottom=False,
                left=False,
                right=False,
            )
            axp.tick_params(
                labelcolor="none",
                which="both",
                top=False,
                bottom=False,
                left=False,
                right=False,
            )
        grid[0].set_ylabel("Ground truth")
        grid[1].set_ylabel("Prediction")
        for i in range(nsamples):
            grid[2 * i].set_title(f"Sample {i}")
        plt.tight_layout()
        tensorboard_writer.add_figure("GT and predicted mask", fig, global_step=e)

        # Update the learning rate with the scheduler policy
        scheduler.step(macro_test_F1)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    license = """
    main.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    logging.info(license)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test"])

    parser.add_argument("--logdir", type=pathlib.Path, default="./logs")
    parser.add_argument("--commit_id", type=str, default=None)
    parser.add_argument("--datadir", type=pathlib.Path, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", choices=models.available_models, required=True)

    # Training parameters
    parser.add_argument("--logname", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    exec(f"{args.command}(args)")
