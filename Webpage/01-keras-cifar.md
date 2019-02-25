---
title:  'A more ambitious image classification dataset : CIFAR-100'
author:
- Jeremy Fix
keywords: [Keras tutorial, CIFAR-100]
...



## Objectives

We now turn to a more difficult problem of classifying RBG images belonging to one of 100 classes with the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 50000 training images and 10000 test images. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs). To give you an idea of the coarse labels, you find fruits, fish, aquatic mammals, vehicles, ... and the fine labels are for example seal, whale, orchids, bicycle, bus, ... Keras provides [functions to automatically get the CIFAR-100 dataset](https://keras.io/datasets/).

Classical dataset augmentation in CIFAR-100 include :

- feature wise standardization
- horizontal flip
- zero padding of 4 pixels on each side, with random crops of 32x32.

For the last augmentation, you can make use of width_shift_range, height_shift_range, fill_mode="constant" and cval=0.

I now propose you a list of recent papers published on arxiv and I propose you to try reimplementing their architecture and training setup :

- MobileNet [@Howard2017] [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Application](https://arxiv.org/pdf/1704.04861.pdf)
- SqueezeNet [@Iandola2016] [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and 0.5M model size](https://arxiv.org/pdf/1602.07360.pdf)
- DenseNet [@Huang2017] [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- WideResNet [@Zagoruyko2016] [Wide Residual Networks](https://arxiv.org/pdf/1605.07146.pdf)
- Xception [@Chollet2016] [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357v3.pdf)
- NASNet : [@Zoph2017] [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/pdf/1707.07012.pdf)

The following papers are trickier to implement :

- ShuffleNet [@Zhang2017] [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)
- ResNet with Stochastic Depth [@Huang2016] [Deep networks with stochastic depth](https://arxiv.org/pdf/1603.09382.pdf)
- Shake-Shake [@Gastaldi2017] [Shake-Shake regularization](https://arxiv.org/pdf/1705.07485.pdf)

If you wish to get an idea of the state of the art in 2015 on CIFAR-100, I invite you to visite the [classification scores website](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).

## References
