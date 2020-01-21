#!/usr/bin/env python3

# Standard modules
import os
# External modules
import torch
import torch.nn as nn
from ray import tune
# Local modules
import utils
import models
import data
import train


def get_optimizer(config, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    return optimizer


def train_tune(config):
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device {}".format(device))
    train_loader, valid_loader, _, _, _ = data.get_data_loaders(config)
    model = models.get_model(config)
    model = model.to(device)

    loss = nn.CrossEntropyLoss()  # This computes softmax internally
    optimizer = get_optimizer(config, model)

    while True:
        train_loss, train_acc = utils.train(model,
                                            train_loader,
                                            loss,
                                            optimizer,
                                            device,
                                            verbose=False)
        val_loss, val_acc = utils.test(model, valid_loader, loss, device)
        tune.track.log(mean_accuracy=val_acc, mean_loss=val_loss)


def merge_configs(config, tunable_config):
    ccopy = config.copy()
    ccopy.update(tunable_config)
    return ccopy


if __name__ == '__main__':

    # Disable proxy, causes issues with ray[tune]
    os.environ.pop('http_proxy')
    os.environ.pop('https_proxy')

    config = train.parse_args()
    config.pop('lr')
    tunable_config = {
        'lr': tune.grid_search([0.001, 0.01, 0.1])
    }
    if config['use_gpu']:
        resources_per_trial = {"gpu": 1,
                               "cpu": 7}
    else:
        resources_per_trial = {"cpu": 7}

    analysis = tune.run(lambda tc: train_tune(merge_configs(config, tc)),
                        resources_per_trial=resources_per_trial,
                        config=tunable_config)
