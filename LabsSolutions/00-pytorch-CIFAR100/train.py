
import torch
import torch.optim
import torch.nn as nn
import torchvision.transforms as transforms
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
        '--normalize',
        action='store_true',
        help='Whether to normalize the dataset'
    )
    parser.add_argument(
        '--dropout',
        action='store_true',
        help='Whether to use dropout'
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
        choices=['linear', 'cnn', 'wrn', 'wide'],
        action='store',
        required=True
    )

    args = parser.parse_args()

    epochs      = 200
    valid_ratio = 0.2
    batch_size  = 128
    weight_decay= args.weight_decay
    num_workers = args.num_workers
    dataset_dir = args.dataset_dir

    if args.data_augment:
        train_augment_transforms = [transforms.RandomHorizontalFlip(0.5), transforms.Pad(4), transforms.RandomCrop((32, 32))]
    else:
        train_augment_transforms = []

    normalize   = args.normalize
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
                                                normalize,
                                                dataset_dir,
                                                train_augment_transforms)
    # Model definition
    model = models.build_model(args.model, input_dim, num_classes, use_dropout, weight_decay)
    model = model.to(device=device)

    # Loss function
    loss = nn.CrossEntropyLoss()  # This computes softmax internally

    # Optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
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
\t{}

{} trainable parameters

Optimizer
========
\t{}

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
                                             {'model': model,
                                              'normalization_function': normalization_function})

    # Training loop
    for t in range(epochs):
        scheduler.step()

        print("Epoch {}".format(t))
        train_loss, train_acc = utils.train(model, train_loader, loss, optimizer, device)

        val_loss, val_acc = utils.test(model, valid_loader, loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))

        model_checkpoint.update(val_loss)
        tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
        tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, t)
        tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
        tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, t)


