
import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import RandomAffine
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning as pl

import numpy as np

import models
import data

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Where to store the downloaded dataset',
        default=None
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay'
    )

    parser.add_argument(
        '--data_augment',
        help='Specify if you want to use data augmentation',
        action='store_true'
    )

    parser.add_argument(
        '--normalize',
        help='Which normalization to apply to the input data',
        action='store_true'
    )

    parser.add_argument(
        '--logdir',
        type=str,
        default="./logs",
        help='The directory in which to store the logs'
    )

    parser.add_argument(
        '--model',
        choices=['linear', 'fc', 'fcreg', 'vanilla', 'fancyCNN'],
        action='store',
        required=True
    )


    args = parser.parse_args()

    img_width = 28
    img_height = 28
    img_size = (1, img_height, img_width)
    num_classes = 10
    batch_size=128
    epochs=10
    valid_ratio=0.2

    # FashionMNIST dataset
    train_augment_transforms = None
    if args.data_augment:
        train_augment_transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                       RandomAffine(degrees=10, translate=(0.1, 0.1))])


    train_loader, valid_loader, test_loader, normalization_function = data.load_fashion_mnist(valid_ratio,
                                                                                              batch_size,
                                                                                              args.num_workers,
                                                                                              args.normalize,
                                                                                              dataset_dir = args.dataset_dir,
                                                                                              train_augment_transforms = train_augment_transforms)



    # Init model, loss, optimizer
    model = models.build_model(args.model, img_size, num_classes)

    class LitModel(pl.LightningModule):

        def __init__(self,
                     weight_decay,
                     model: nn.Module = None) -> None:
            super().__init__()
            self.weight_decay = weight_decay
            self.model = model
            self.loss = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.model(x)

        def configure_optimizers(self):
            #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=self.weight_decay)
            return optimizer

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            return self.loss(y_hat, y)
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            return self.loss(y_hat, y)

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            return self.loss(y_hat, y)
       
    lmodel = LitModel(args.weight_decay, model)

    trainer = pl.Trainer()  
    trainer.fit(lmodel, train_loader, valid_loader)
