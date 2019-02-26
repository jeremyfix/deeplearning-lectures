import sys

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Lambda
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import h5py
import numpy as np

from keras.callbacks import TensorBoard, ModelCheckpoint

import models

parser = argparse.ArgumentParser()


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

# Loading the MNIST dataset
# For MNIST, input_shape is (28, 28). The images are monochrome
(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_classes = 10
img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
num_channels = 1
num_train = X_train.shape[0]
num_test = X_test.shape[0]
input_shape = (img_rows, img_cols, num_channels)

def split(X, y, test_size):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    nb_test = int(test_size * X.shape[0])
    return X[nb_test:,:, :], y[nb_test:],\
           X[:nb_test, :, :], y[:nb_test]


X_train, y_train, X_val, y_val = split(X_train, y_train, test_size=0.1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
y_train = to_categorical(y_train, num_classes)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
y_val = to_categorical(y_val, num_classes)

X_test = X_test.reshape(num_test, img_rows, img_cols, 1)
y_test = to_categorical(y_test, num_classes)

normalization = None
if args.normalize:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0 # Do not modify points where variance is null
    normalization = lambda xi: Lambda(lambda image, mu, std: (image - mu) / std,
	       		arguments={'mu': mean, 'std': std})(xi)



# We build the requested model
model = models.build_network(args.model, input_shape, num_classes,
		  	     normalization, args.dropout, args.L2)
model.summary()

# Callbacks

def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "-" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

logpath = generate_unique_logpath(args.logdir, args.model)
tbcb = TensorBoard(log_dir=logpath)
print("=" * 20)
print("The logs will be saved in {}".format(logpath))
print("=" * 20)

checkpoint_filepath = os.path.join(logpath,  "best_model.h5")
checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)

# Compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training
if args.data_augment:
    datagen = ImageDataGenerator(shear_range=0.3,
			      zoom_range=0.1,
			      rotation_range=10.)

    train_flow = datagen.flow(X_train, y_train, batch_size=128)
    history = model.fit_generator(train_flow,
			steps_per_epoch=X_train.shape[0]/128,
			epochs=50,
			verbose=1,
			validation_data = (X_val, y_val),
			callbacks=[tbcb, checkpoint_cb])

else:
    history = model.fit(X_train, y_train,
			batch_size=128,
			epochs=50,
			verbose=1,
			validation_data = (X_val, y_val),
			callbacks=[tbcb, checkpoint_cb])

with h5py.File(checkpoint_filepath, 'a') as f:
    if 'optimizer_weights' in f.keys():
        del f['optimizer_weights']

# Evaluation of the best model
model = load_model(checkpoint_filepath)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
