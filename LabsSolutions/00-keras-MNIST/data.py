#!/usr/bin/env python3

# External imports
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def split(X, y, test_size):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    nb_test = int(test_size * X.shape[0])
    return X[nb_test:, :, :], y[nb_test:], X[:nb_test, :, :], y[:nb_test]

def get_data(normalize: bool,
             augment: bool):
    """
    Loads the dataset and compute the normalization data and possibly
    prepare data augmentation
    """

    # Loading the MNIST dataset
    # For MNIST, input_shape is (28, 28). The images are monochrome
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    num_classes = 10
    img_rows = X_train.shape[1]
    img_cols = X_train.shape[2]
    num_channels = 1
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    input_shape = (img_rows, img_cols, num_channels)

    X_train, y_train, X_val, y_val = split(X_train, y_train, test_size=0.1)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    y_train = to_categorical(y_train, num_classes)

    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
    y_val = to_categorical(y_val, num_classes)

    X_test = X_test.reshape(num_test, img_rows, img_cols, 1)
    y_test = to_categorical(y_test, num_classes)

    normalization = None
    if normalize:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0 # Do not modify points where variance is null
        normalization = (mean, std)

    traindata = (X_train, y_train)
    if augment:
        datagen = ImageDataGenerator(shear_range=0.3,
                                     zoom_range=0.1,
                                     rotation_range=10.)

        traindata = datagen.flow(*traindata,
                                 batch_size=128)

    return traindata, (X_val, y_val), (X_test, y_test), normalization, input_shape, num_classes


if __name__ == '__main__':
    pass
