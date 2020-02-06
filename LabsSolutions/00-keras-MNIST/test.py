
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import argparse
import sys
import os
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    help='Model to test'
)

parser.add_argument(
    '--modelsdir',
    type=str,
    help='Directory where to look for models to test'
)

args = parser.parse_args()
if(not args.model and not args.modelsdir):
    print("--model or --modelsdir is required. Call with -h for help")
    sys.exit(-1)

# Loading the MNIST dataset
# For MNIST, input_shape is (28, 28). The images are monochrome
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_classes = 10
img_rows = X_test.shape[1]
img_cols = X_test.shape[2]
num_channels = 1
num_test = X_test.shape[0]
y_test_cat = to_categorical(y_test, num_classes)


def clean_h5(model_file):
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']

def test_loop(modelpath):
    '''
    Highly inefficient, obviously not advised
    But it shows how to compute the labels of images
    '''
    model = load_model(modelpath)

    Xt = X_test.reshape(num_test, img_rows, img_cols, num_channels).astype('float32')
    ypred = np.zeros((num_test, ))
    ypred_label = np.zeros((num_test, ))
    for i in range(Xt.shape[0]):
        y = model.predict(Xt[i, ...].reshape((1, img_rows, img_cols, 1)))
        ypred[i] = y[0][y_test[i]]
        ypred_label[i] = y[0].argmax()

    print('Model {}'.format(modelpath))
    print('  Test loss:', -np.log(ypred).mean())
    print('  Test accuracy:', (ypred_label == y_test).mean())



def test_model(modelpath):
    # We now load the model
    clean_h5(modelpath)
    model = load_model(modelpath)

    Xt = X_test.reshape(num_test, img_rows, img_cols, num_channels)

    score = model.evaluate(Xt, y_test_cat, verbose=0)
    print('Model {}'.format(modelpath))
    print('  Test loss:', score[0])
    print('  Test accuracy:', score[1])


if args.model:
    test_model(args.model)
    test_loop(args.model)
elif args.modelsdir:
    for root, dirs, files in os.walk(args.modelsdir):
        if("best_model.h5" in files):
            test_model(os.path.join(root, "best_model.h5"))
