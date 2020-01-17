
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

import numpy as np

import utils
import models
import data

def train(config):
    '''
    Run a training with the provided parameters

    config (dict):
        use_gpu : True/False
        log : None or dict
        model : linear, fc, fcreg, vanilla, fancyCNN
    '''
    img_width = 28
    img_height = 28
    img_size = (1, img_height, img_width)
    num_classes = 10
    batch_size=128
    epochs=10
    valid_ratio=0.2
    tensorlogs = config['log'] is not None

    if config['use_gpu']:
        print("Using GPU{}".format(torch.cuda.current_device()))
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')


    # Where to store the logs
    if tensorlogs: 
        logdir = utils.generate_unique_logpath(config['log']['logdir'], 
                                               config['model'])
        print("Logging to {}".format(logdir))
        if not os.path.exists(config['log']['logdir']):
            os.mkdir(config['log']['logdir'])
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    # FashionMNIST dataset
    train_augment_transforms = None
    if args.data_augment:
        train_augment_transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                       RandomAffine(degrees=10, translate=(0.1, 0.1))])


    train_loader, valid_loader, test_loader, normalization_function = data.load_fashion_mnist(valid_ratio,
                                                                                              batch_size,
                                                                                              config['num_workers'],
                                                                                              config['normalize'],
                                                                                              dataset_dir =
                                                                                              config['dataset_dir'],
                                                                                              train_augment_transforms = train_augment_transforms)



    # Init model, loss, optimizer
    model = models.build_model(args.model, img_size, num_classes)
    model = model.to(device)

    loss = nn.CrossEntropyLoss()  # This computes softmax internally
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)

    # Where to save the logs of the metrics
    history_file = open(logdir + '/history', 'w', 1)
    history_file.write("Epoch\tTrain loss\tTrain acc\tVal loss\tVal acc\tTest loss\tTest acc\n")

    # Generate and dump the summary of the model
    model_summary = utils.torch_summarize(model)
    print("Summary:\n {}".format(model_summary))

    summary_file = open(logdir + "/summary.txt", 'w')
    summary_text = """

Executed command
===============
{}

Dataset
=======
Train transform : {}
Normalization : {}

Model summary
=============
{}

{} trainable parameters

Optimizer
========
{}

    """.format(" ".join(sys.argv),
               train_augment_transforms,
               args.normalize,
               str(model).replace('\n','\n\t'),
               sum(p.numel() for p in model.parameters() if p.requires_grad),
               str(optimizer).replace('\n', '\n\t'))
    summary_file.write(summary_text)
    summary_file.close()


    tensorboard_writer   = SummaryWriter(log_dir = logdir)
    tensorboard_writer.add_text("Experiment summary", summary_text)
    model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt",
                                             {'model': model, 'normalization_function': normalization_function}
                                            )
    # Add the graph of the model to the tensorboard
    inputs, _ = next(iter(train_loader))
    inputs = inputs.to(device)
    tensorboard_writer.add_graph(model, inputs)
    ####################################################################################### Main Loop
    for t in range(epochs):
        print("Epoch {}".format(t))
        train_loss, train_acc = utils.train(model, train_loader, loss, optimizer, device)

        val_loss, val_acc = utils.test(model, valid_loader, loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))

        test_loss, test_acc = utils.test(model, test_loader, loss, device)
        print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))

        history_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(t,
                                                                 train_loss, train_acc,
                                                                 val_loss, val_acc,
                                                                 test_loss, test_acc))
        model_checkpoint.update(val_loss)
        tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
        tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, t)
        tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
        tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, t)
        tensorboard_writer.add_scalar('metrics/test_loss', test_loss, t)
        tensorboard_writer.add_scalar('metrics/test_acc',  test_acc, t)


    # Loading the best model found

    print("Loading and testing the best model")

    best_model_path = os.path.join(config['logdir'], "best_model.pt")
    loaded_dict = torch.load(best_model_path)
    model = loaded_dict['model'].to(device)

    model.eval()

    val_loss, val_acc = utils.test(model, valid_loader, loss, device)
    print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))

    test_loss, test_acc = utils.test(model, test_loader, loss, device)
    print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))


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

    config = {
        'use_gpu': args.use_gpu,
        'log' : {'logdir': args.logdir},
        'model': args.model,
        'num_workers': args.num_workers,
        'normalize': args.normalize,
        'dataset_dir': args.dataset_dir
    }


    train(config)
