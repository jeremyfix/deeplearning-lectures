---
title:  'Semantic segmentation'
author:
- Jeremy Fix
keywords: [PyTorch tutorial, semantic segmentation, Stanford 2D-3D S]
...

## Objectives

The objective of this lab work is to implement and explore convolutional neural networks for semantic segmentation. **Semantic segmentation** seeks to learn a function $f$, parametrized by $\theta$ which takes as input a colored image $I$ of arbitrary shape $H\times W$ and outputs an images of labels of the same shape $H \times W$ than the input. Indeed, we seek to label every single pixel of the image as belonging to one of $K$ predefined classes.

In this labwork, we will be working with the large [Stanford 2D-3D S dataset](http://buildingparser.stanford.edu/dataset.html). This dataset is built from 3D scans of buildings with multiple annotation types (pixelwise depth, pixel class, pixelwise normals, scene category). We will consider only the pixelwise class labeling. The data have been collected in 6 different areas. 

| Area | Number of images|
| ---  | --- |
| 1    | 10,327 |
| 2    | 15,714 |
| 3    | 3,704 |
| 4    | 13,268 |
| 5    | 17,593 |
| 6    | 9,890 |
| **Total** | 25,434 |

Below is an example of the input RGB image and the associated labels 

![Semantic segmentation : pixelwise classifcation](./data/01-pytorch-segmentation/sample.png){width=50%}

There are $14$ classes, the first being for the unlabeled pixels.

## Setup and predefined scripts

For this lab work, you are provided with environment setup files, either a [environment.yml](./data/01-pytorch-segmentation/environment.yml) if you are using conda and a [requirements.txt](./data/01-pytorch-segmentation/requirements.txt) file if you are a pipenv/pip/virtualenv/etc..

On the CentraleSupelec clusters, I advise you to use conda :

``` console
sh11:~:mylogin$ export PATH=PATH:/opt/conda/bin
sh11:~:mylogin$ conda env create -f environment.yml
sh11:~:mylogin$ source activate dl-lectures-pytorch-segmentation
```

If your enviromnent is correctly created and activated, the following should work 

```console
(dl-lectures-pytorch-segmentation) sh11:~:mylogin$ python -c "import torch; print(torch.__version__)"
1.10.1+cu113
```

Your are also provided with some base code :

- [data.py](./labs/01-pytorch-segmentation/data.py) : script responsible for dataset and dataloader building
- [models.py](./labs/01-pytorch-segmentation/models.py) : script holding the models zoo
- [metrics.py](./labs/01-pytorch-segmentation/metrics.py) : script for the definition of custom metrics
- [main.py](./labs/01-pytorch-segmentation/main.py) : the main script orchestrating the call to all your functions for training and inference
- [utils.py](./labs/01-pytorch-segmentation/utils.py) : several utilitary functions for colorize segmentation masks
- [test_implementation.py](./labs/01-pytorch-segmentation/test_implementation.py) : a script with unit tests to test your answers to the code questions



## Data exploration

TODO: ask them to load a minibatch check the input tensors, target tensors shape and content, plot a sample using the provided functions colorize, etc..

TODO: provide ids of images which have incorrect labels, e.g.
area_5a/data/semantic/camera_ff5f377af3b34354b054536db27565ae_hallway_7_frame_4_domain_semantic.png

## Data pipeline

The data are loaded and the dataloaders are built by calling `get_dataloaders` from the `data` module.

The raw images have a size of $1080 \times 1080$. If we keep large images, the minibatches and their successive transformations will occupy a large space in GPU memory. At least, we can resize the images, e.g. to $256\times 256$, and still keep reasonnable segmentations.

In addition to resize images to a smaller size, we can add data augmentation in our pipeline. Remember, data augmentation is a set of transforms applied to the input which have a predictable influence on the output.

For semantic segmentation, several can be considered, such as [RandomCrop](https://albumentations.ai/docs/api_reference/augmentations/crops/transforms/#albumentations.augmentations.crops.transforms.RandomCrop), [HorizontalFlip](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.HorizontalFlip), [MaskDropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/mask_dropout/#maskdropout-augmentation-augmentationsdropoutmask_dropout), [CoarseDropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#coarsedropout-augmentation-augmentationsdropoutcoarse_dropout). You can have a look at the list of transforms provided by albumentation [here](https://albumentations.ai/docs/getting_started/transforms_and_targets/).

<div class="w3-card w3-sand">
The transforms are organized into modules with albumentations, e.g. [CoarseDropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/#coarsedropout-augmentation-augmentationsdropoutcoarse_dropout) is defined in `albumentations.augmentations.dropout.coarse_dropout.CoarseDropout`. However, the code of albumentations imports all the classes of the submodules recursively on the parent `__init__.py` script. Hence, the coarse dropout (and all the others) can be simply created by calling `A.CoarseDropout()` if you `import albumentations as A`.
</div>

In the provided code, the data loading pipeline simply :

- scales the image by $255$ with [albumentation.Normalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize)
- transforms the numpy image into a pytorch tensor with [albumentation.ToTensorV2](https://albumentations.ai/docs/api_reference/pytorch/transforms/#albumentations.pytorch.transforms.ToTensorV2) 
 
**Question** In the `train` function of the main script, implement the data augmentation operations you identified as relevant. For testing visually your data augmentation pipeline, you can implement your data augmentation pipeline in the `data.py:test_augmented_dataset` function and evaluate the `data.py` script to see the results. Once satisfied, you can inject your augmentations in the `main.py` script.

**Question** How would you quantify the quality of the hyperparameters of your augmentation pipeline ? 

## Model implementation

The parametric model you learn takes as input a 3-channel image and outputs a probability distribution over the $14$
classes. There has several several propositions in the litterature to adress this problem such as FCN [@Long2015], UNet [@Ronneberger2015], VNet [@Milletari2016], SegNet[@Badrinarayanan2017], DeepLab v3+ [@Chen2018]. In this labwork, I propose we code the UNet of 2015 and you might want to implemented DeepLabv3+ as a homework :)

UNet is a fully convolutional network (FCN), i.e. involving only convolutional operations (Conv2D, MaxPool , ...; there is no fully connected layers). 

**Question** What does it mean that UNet is a fully convolutional network ? What does it mean with respect to the input image sizes it can process ? 

**Question** For minibatch training, what is the constraint that we have on the input image sizes ?

The name U-Net comes from the very specific shape of this encoder-decoder network with a contracting pathway for the
encoding following by an expanding pathway for the decoding. The contracting pathway is expected to learn higher and
higher level features as we progress deeper in the network and the expanding pathway to merge these highlevel features
with the finer grain details brought by the encoder through skip layer connections.

![UNet architecture for semantic segmentation](./latex/unet.png){width=50%}


The provided code implements `UNet` with a `UNetEncoder` class and a `UNetDecoder` class. Both the UNetEncoder and
UNetDecoder relies on repetition of blocks which are UNetConvBlock on the one hand and UNetUpConvBlock on the other
hand.

**UNetEncoder** This downsampling pathway is made of :

- several `UNetConvBlock` blocks where the $i$-th ($i \in [0, \#blocks-1]$) block is made of:
	- `block1`=Conv($3\times 3$)-ReLU-BatchNorm, with $64 \times 2^i$ channels 
	- `block2`=Conv($3\times 3$)-ReLU-BatchNorm, with $64 \times 2^i$ channels and 
	- `block3`= MaxPooling($2\times 2$),
- followed by a Conv($3\times 3$)-ReLU-BatchNorm, with $64 \times 2^{\#blocks}$ channels

Note that the output of `block2`, just before the downsampling is transmitted to the decoder stage, therefore the `UNetConvBlock` forward function outputs two tensors : one to be propagated along the encoder and one to be transmitter to the decoder.

**Question** In the `models.py` script, implement the code in the `UNetEncoder` and `UNetConvBlock` classes. Check your code running the unit tests. Be sure to understand how the propagation is performed through the encoder blocks.

The output of the final layers of the encoder is a tensor of shape $(batch, 64\times 2^{\#blocks}, H/2^{\#blocks}, W/2^{\#blocks})$

**UNetDecoder** This upsampling pathway is made of :

- a first Conv($3\times 3$)-ReLU-BatchNorm block which keeps constant the number of channels,
- followed by several `UNetUpConvBlock` blocks with :
	- `upconv`=UpSample($2$)-Conv($3\times 3$)-ReLU-BatchNorm which halves the number of channels, 
	- a concatenation of the upconv output features with the encoder features
	- a `convblock`= Conv($3\times 3$)-ReLU-BatchNorm-Conv($3\times 3$)-ReLU-BatchNorm which halves the number of its input channels

For the `UNetUpConvBlock`, when its input along the decoder pathway has $c_0$ channels, its output has $c_0/2$ channels since :

- upconv outputs $c_0/2$ channels
- the concatenation of $c_0/2$ channels with the $c_0/2$ channels of the encoder leads to $c_0$ channels
- the `convblock` gets $c_0$ input channels and outputs $c_0/2$ output channels

In order to output a score for each of the $14$ classes, the very last layer of the decoder is a Conv($1\times 1$) with the same number of channels than the number of classes.

**Question** In the `models.py` script, implement the code in the `UNetDecoder` and `UNetUpConvBlock` classes. Check your code against the unit tests. Be sure to understand how the propagation is performed through the decoder blocks.

Once both the encoder and decoder classes are implemented, you can see that the `UNet` class is a simple wrapper around them.

**Question** In the `models.py` script, within the `main()` function, write a piece of code for propagating a dummy tensor within your model. Is the output tensor the shape you expect ?

We are done with the model implementation, let us move on to the loss.

## Loss implementation



## Evaluation metrics

## Training on a small subset

TODO: ask them to train and monitor the training both qualitatively and quantitatively on a small subset of the data

TODO : give an idea of the training time over the whole data

## Inference

TODO: provide a pretrained network on the whole dataset

## Going further

### Models

### Losses

## A possible solution

## References
