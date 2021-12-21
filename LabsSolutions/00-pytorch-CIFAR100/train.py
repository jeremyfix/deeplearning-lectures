# coding: utf-8

# Standard imports
import argparse
import os
import sys

# External imports
import torch
import torch.optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import deepcs
import deepcs.display
from deepcs.training import train, ModelCheckpoint
from deepcs.testing import test
from deepcs.fileutils import generate_unique_logpath
from deepcs.metrics import accuracy

# Local imports
import data
import models
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Whether to use GPU'
    )

    parser.add_argument(
        '--normalization',
        choices=['None', 'img_meanstd', 'channel_meanstd'],
        action='store',
        required=True,
        default='None',
        help='Whether and How to normalize the dataset'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default= 0.0,
        help='Dropout rate (prob to zeros). The default dropout=0.0 disables dropout'
    )
    parser.add_argument(
        '--data_augment',
        action='store_true',
        help='Whether to use dataset augmentation on the training set'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Where to store the downloaded dataset',
        required=True
    )

    parser.add_argument(
        '--logdir',
        type=str,
        help='Where to store the logs',
        default='./logs'
    )

    parser.add_argument(
        '--cyclic_lr',
        nargs=3,
        type=float,
        metavar=('low', 'high', 'period'),
        help='If you want to use cyclic LR'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='The L2 regularization coefficient'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--model',
        choices=['linear', 'cnn', 'wrn', 'wide', 'resnet18'],
        action='store',
        required=True
    )

    args = parser.parse_args()

    epochs      = 200
    valid_ratio = 0.1
    batch_size  = 128
    weight_decay= args.weight_decay
    num_workers = args.num_workers
    dataset_dir = args.dataset_dir

    if args.data_augment:
        train_augment_transforms = [transforms.RandomHorizontalFlip(0.5), transforms.RandomCrop((32, 32), padding=4)]
    else:
        train_augment_transforms = []

    normalization   = args.normalization
    use_dropout = args.dropout

    input_dim = (3, 32, 32)
    num_classes = 100

    if args.use_gpu:
        print("Using GPU{}".format(torch.cuda.current_device()))
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    # Data loading
    train_loader, valid_loader, normalization_function = data.load_data(valid_ratio,
                                                batch_size,
                                                num_workers,
                                                normalization,
                                                dataset_dir,
                                                train_augment_transforms)

    test_loader = data.load_test_data(batch_size,
                                      num_workers,
                                      dataset_dir,
                                      normalization_function)

    # Model definition
    model = models.build_model(args.model, input_dim, num_classes, use_dropout, weight_decay)
    model = model.to(device=device)

    # Loss function
    loss = nn.CrossEntropyLoss()  # This computes softmax internally
    metrics = {
        'CE': loss,
        'accuracy': accuracy
    }

    # Optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if args.cyclic_lr:
        low_lr, high_lr, period = args.cyclic_lr
        def cyclical_lr(epoch, low_lr=low_lr, high=high_lr, period=period):
            dt = (epoch % period) / period
            if dt < 0.5:
                # rising phase
                return low_lr + 2.0 * dt * (high_lr - low_lr)
            else:
                return high_lr + 2.0 * (dt-0.5) * (low_lr - high_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cyclical_lr)
    else:
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120,150], gamma=0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma=0.5)

    # Callbacks

    ## Where to store the logs
    logdir = generate_unique_logpath('./logs', args.model)
    print(f"Logging to {logdir} ")
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)


    # Display information about the model
    summary_text = f"""## Summary of the model architecture

{deepcs.display.torch_summarize(model, (batch_size, ) + input_dim)}

## Executed command

{' '.join(sys.argv)}


## Dataset

CIFAR-100

Normalization : {args.normalization}

Train augmentation : {train_augment_transforms}

## Optimizer

{optimizer}

"""


    print(summary_text)
    with open(logdir + "/summary.txt", 'w') as f:
        f.write(summary_text)

    ## Tensorboard writer
    tensorboard_writer = SummaryWriter(log_dir = logdir)
    tensorboard_writer.add_text("Experiment summary", summary_text)

    ## Checkpoint
    model_checkpoint = ModelCheckpoint(model,
                                       os.path.join(logdir, 'best_model.pt'))
    #TODO: save the normalization_function
    
    # Training loop
    for t in range(epochs):
        print("Epoch {}".format(t))
        train(model, train_loader, loss, optimizer, device, metrics,
              num_epoch=t,
              tensorboard_writer=tensorboard_writer, 
              dynamic_display=True)

        val_metrics = test(model, valid_loader, device, metrics)
        updated = model_checkpoint.update(val_metrics['CE'])
        print("[%d/%d] Valid:   Loss : %.3f | Acc : %.3f%% %s"% (t,
                                                             epochs,
                                                             val_metrics['CE'],
                                                             100.*val_metrics['accuracy'],
                                                             "[>> BETTER <<]" if updated else ""))

        test_metrics = test(model, test_loader, device, metrics)
        print("[%d/%d] Test:    Loss : %.3f | Acc : %.3f%%"% (t,
                                                              epochs,
                                                              test_metrics['CE'],
                                                              100.*test_metrics['accuracy']))

        # Write the metrics to the tensorboard
        for m_name, m_value in val_metrics.items():
            tensorboard_writer.add_scalar(f'metrics/val_{m_name}', m_value, t)
        for m_name, m_value in test_metrics.items():
            tensorboard_writer.add_scalar(f'metrics/test_{m_name}', m_value, t)

        scheduler.step()
