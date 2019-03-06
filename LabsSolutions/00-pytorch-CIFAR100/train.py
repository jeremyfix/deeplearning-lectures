
import torch
import torch.optim
import torch.nn as nn
from tensorboardX import SummaryWriter

import argparse
import os
import sys

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
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--model',
        choices=['linear', 'cnn'],
        action='store',
        required=True
    )

    args = parser.parse_args()

    valid_ratio = 0.2
    batch_size  = 128
    num_workers = args.num_workers
    dataset_dir = args.dataset_dir
    train_augment_transform = []

    input_dim = (3, 32, 32)
    num_classes = 100

    if args.use_gpu:
        print("Using GPU{}".format(torch.cuda.current_device()))
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    # Data loading
    train_loader, valid_loader = data.load_data(valid_ratio,
                                                batch_size,
                                                num_workers,
                                                dataset_dir,
                                                train_augment_transform)
    # Model definition
    model = models.build_model(args.model, input_dim, num_classes)
    model = model.to(device=device)

    # Loss function
    loss = nn.CrossEntropyLoss()  # This computes softmax internally

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Callbacks

    ## Where to store the logs
    logdir = utils.generate_unique_logpath(args.logdir, args.model)
    print("\n Logging to {} \n".format(logdir))
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)


    ## Summary file
    model_summary = utils.torch_summarize(model)
    print("Summary:\n {}".format(model_summary))

    summary_file = open(logdir + "/summary.txt", 'w')
    summary_text = """

Executed command
===============
{}

Dataset
=======
CIFAR-100

Model summary
=============
{}

{} trainable parameters

Optimizer
========
{}

    """.format(" ".join(sys.argv),
            str(model).replace('\n','\n\t'),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            str(optimizer).replace('\n', '\n\t'))

    summary_file.write(summary_text)
    summary_file.close()

    ## Tensorboard writer
    tensorboard_writer   = SummaryWriter(log_dir = logdir)
    tensorboard_writer.add_text("Experiment summary", summary_text)

    ## Checkpoint
    model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt",
                                             {'model': model})

    # Training loop


