#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import argparse
import sys
import os
import functools
# External imports
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import h5py
import numpy as np
# Local imports
import data
import models


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "-" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
            return log_path
        i = i + 1


def train(args):
    """
    Training of the algorithm
    """

    train_data, val_data, test_data, normalization, input_shape, num_classes = data.get_data(args.normalize,
                                                                                   args.data_augment)

    # We build the requested model
    model = models.build_network(args.model,
                                 input_shape,
                                 num_classes,
                                 normalization,
                                 args.dropout,
                                 args.L2)
    model.summary()

    # Callbacks
    logpath = generate_unique_logpath(args.logdir, args.model)
    tbcb = TensorBoard(log_dir=logpath)

    print("=" * 20)
    print("The logs will be saved in {}".format(logpath))
    print("=" * 20)

    checkpoint_filepath = os.path.join(logpath,  "best_model.h5")
    checkpoint_cb = ModelCheckpoint(checkpoint_filepath,
                                    save_best_only=True)

    # Write down the summary of the experiment
    with open(os.path.join(logpath, "summary.txt"), 'w') as f:
        f.write("## Executed command \n")
        f.write(" ".join(sys.argv) + '\n')
        f.write("\n## Args\n"
                f"{args}""")

    # Compilation
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Training
    if args.data_augment:
        ffit = functools.partial(model.fit,
                                 train_data,
                                 steps_per_epoch=50000//128)
    else:
        ffit = functools.partial(model.fit,
                                 *train_data,
                                 batch_size=128)

    ffit(epochs=50,
         verbose=1,
         validation_data=val_data,
         callbacks=[tbcb, checkpoint_cb])

    with h5py.File(checkpoint_filepath, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']

    # Evaluation of the best model
    model = load_model(checkpoint_filepath)
    score = model.evaluate(*test_data, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("command",
                        choices=['train', 'test'])

    parser.add_argument(
        '--logdir',
        type=str,
        default="./logs/",
        help='The directory in which to store the logs (default: ./logs_fc)'
    )

    parser.add_argument(
        '--normalize',
        help='Should we standardize the input data',
        action='store_true'
    )

    parser.add_argument(
        '--data_augment',
        help='Whether to use data augmentation',
        action='store_true'
    )

    parser.add_argument(
        '--L2',
        help='Whether to use L2 regularization (applies for FC and CNN)',
        action='store_true'
    )
    parser.add_argument(
        '--dropout',
        help='Whether to use Dropout regularization (applies for FC and CNN)',
        action='store_true'
    )

    parser.add_argument(
        '--model',
        choices=['linear', 'fc', 'vanilla', 'fancy'],
        action='store',
        required=True
    )

    args = parser.parse_args()

    eval(f"{args.command}(args)")
