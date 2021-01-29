#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import argparse
import sys
import os
import functools
# External imports
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import h5py
import numpy as np
import matplotlib.pyplot as plt
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

    if(not args.model):
        print("--model is required for training. Call with -h for help")
        sys.exit(-1)

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


def testimg(args):
    '''
    Test a model or multiple models on an image
    '''
    if(not args.modelpath and not args.modelsdir):
        raise RuntimeError("--modelpath or --modelsdir is required. Call with -h for help")

    if not args.image:
        raise RuntimeError("An image must be given with --image")

    if not args.image:
        X_test, y_test = data.get_test_data()
    else:
        img = np.asarray(Image.open(args.image))
        img = img.reshape((1, 28, 28, 1))

    def test_one_model(modelpath):
        model = load_model(modelpath)
        scores = model.predict(img)
        pred_class = scores.argmax(axis=1)[0]
        pred_proba = scores[0, pred_class]
        return (pred_class, pred_proba)

    if args.modelpath:
        pred_class, pred_proba = test_one_model(args.modelpath)
        print(f"The image is predicted of class {pred_class} with probability {pred_proba}")
    else:
        for root, dirs, files in os.walk(args.modelsdir):
            if("best_model.h5" in files):
                modelpath = os.path.join(root, "best_model.h5")
                pred_class, pred_proba = test_one_model(modelpath)
                print('Model {}'.format(modelpath))
                print(f"The image is predicted of class {pred_class} with probability {pred_proba}")


def test(args):
    '''
    Test a model on the test data
    '''
    if(not args.modelpath and not args.modelsdir):
        raise RuntimeError("--modelpath or --modelsdir is required. Call with -h for help")

    X_test, y_test = data.get_test_data()

    # Code for saving one image from the test set
    # idx = 0
    # xtesti = X_test[idx].squeeze().astype(np.uint8)
    # print(xtesti[0])
    # print(y_test[idx])
    # Image.fromarray(xtesti).save("digit.png")

    def test_one_model(modelpath):
        model = load_model(modelpath)
        score = model.evaluate(X_test, y_test, verbose=0)
        # score = test_loop(model, X_test, y_test)
        return score

    if args.modelpath:
        score = test_one_model(args.modelpath)
        print('Model {}'.format(args.modelpath))
        print('  Test loss:', score[0])
        print('  Test accuracy:', score[1])
    else:
        for root, dirs, files in os.walk(args.modelsdir):
            if("best_model.h5" in files):
                modelpath = os.path.join(root, "best_model.h5")
                score = test_one_model(modelpath)
                print('Model {}'.format(modelpath))
                print('  Test loss:', score[0])
                print('  Test accuracy:', score[1])


def test_loop(model, X_test, y_test):
    '''
    Highly inefficient, obviously not advised
    But it shows how to compute the labels of images

    X_test : [B, H, W, C]
    y_test : one-hot : [B, K]
    '''
    num_test = y_test.shape[0]
    ypred = np.zeros((num_test, ))
    ypred_label = np.zeros((num_test, ))
    ytest_label = y_test.argmax(axis=1)

    for i, (Xtesti, ytesti) in enumerate(zip(X_test, ytest_label)):
        y = model.predict(Xtesti[np.newaxis,...])
        ypred[i] = y[0][ytesti]
        ypred_label[i] = y.argmax(axis=1)

    bce = -np.log(ypred).mean()
    acc = (ypred_label == ytest_label).mean()
    return (bce, acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("command",
                        choices=['train', 'test', 'testimg'])

    # For training
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
        default=None
    )

    # For testing
    parser.add_argument(
        '--modelpath',
        type=str,
        help='Model to test'
    )

    parser.add_argument(
        '--modelsdir',
        type=str,
        help='Directory where to look for models to test'
    )

    parser.add_argument(
        '--image',
        type=str,
        help='On which image to test (path)',
        default=None
    )

    args = parser.parse_args()

    eval(f"{args.command}(args)")
