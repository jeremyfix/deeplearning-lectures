# coding: utf-8

# Standard imports
import argparse
import os
import sys
import glob
# External imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import RandomAffine
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
# Local imports
import models
import data

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
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.01,
                                     weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).sum() / y.shape[0]
        # Note: the log function automatically reduces the stored metrics
        # with its reduce_fx (default: mean)
        # Which means the value is not correct if not all the minibatches
        # have the same size
        self.log('train/loss', loss.detach(), prog_bar=True)
        self.log('train/acc', acc.detach(), prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).sum() / y.shape[0]
        self.log('valid/loss', loss.detach(), prog_bar=True)
        self.log('valid/acc', acc.detach(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).sum() / y.shape[0]
        self.log('test/loss', loss.detach(), prog_bar=True)
        self.log('test/acc', acc.detach(), prog_bar=True)
        return loss

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
        default=7,
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
        '--model',
        choices=['linear', 'fc', 'fcreg', 'vanilla', 'fancyCNN'],
        action='store',
        required=True
    )

    args = parser.parse_args()
    

    loggers = []
    if 'NEPTUNE_TOKEN' in os.environ and 'NEPTUNE_PROJECT' in os.environ:
        print('Using Neptune.ai logger')
        neptune_logger = NeptuneLogger(
            api_key=os.environ['NEPTUNE_TOKEN'],
            project=os.environ['NEPTUNE_PROJECT'],
            tags=["fashionMNIST"],
            source_files=glob.glob('*.py')
        )
        loggers.append(neptune_logger)

    img_width = 28
    img_height = 28
    img_size = (1, img_height, img_width)
    num_classes = 10
    batch_size=128
    epochs = 50
    valid_ratio=0.2

    # Load the dataloaders
    loaders, fnorm = data.make_dataloaders(valid_ratio,
                                           batch_size,
                                           args.num_workers,
                                           args.normalize,
                                           args.data_augment,
                                           args.dataset_dir,
                                           None)
    train_loader, valid_loader, test_loader = loaders

    # Init model, loss, optimizer
    model = models.build_model(args.model,
                               img_size,
                               num_classes)
 
    lmodel = LitModel(args.weight_decay,
                      model)

    trainer = pl.Trainer(max_epochs=epochs,
                         logger=loggers)
    # Train the model on the training set and record the best
    # from the validation set
    trainer.fit(lmodel,
                train_loader,
                valid_loader)

    # And test the best model on the test set
    metrics = trainer.test(dataloaders=test_loader,
                           ckpt_path='best')
