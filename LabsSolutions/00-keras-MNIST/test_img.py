
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt


import argparse
import sys
import os
import h5py

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='Model to test'
)

parser.add_argument(
    '--image',
    type=str,
    required=True,
    help='On which image to test (path)'
)

args = parser.parse_args()

img = plt.imread(args.image)
img = (255 - img.reshape((1, 28, 28, 1)))


model = load_model(args.model)
y = model.predict(img)

print("I think your image {} is a {} with probability {}".format(args.image, y.argmax(), y[0][y.argmax()]))
