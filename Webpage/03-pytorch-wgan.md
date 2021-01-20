---
title:  'Generative Networks (WGAN)'
author:
- Jeremy Fix
keywords: [PyTorch tutorial, WGAN, generative network, MNIST]
...

## Objectives

In this labwork, we aim at experimenting with generative networks and in particular the recently introduced Generative Adversial Networks [@Goodfellow2014]. Although other neural network architectures exist for learning to generate synthetic data from real observations (see for example this [OpenAI blogpost](https://openai.com/blog/generative-models/) which mentions some), the recently introduced GANs framework has shown to be efficient for generating a wide variety of data. 

A GAN network is built from actually two networks that play a two player game :

- a generator which tries to generate images as real as possible, hopefully fooling the second player,
- a critic which tries to distinguish the real images from the fake images

The loss used for training these two neural networks reflect the objective of the generator to fool the critic and of the critic to correctly separate the real from the fake.

The generator generators an image from an input random seed, $z$, say drawn from a normal distribution $\mathcal{N}(0, 1)$. Let us denote $\mathcal{G}(z)$ the output image (for now, we slightly postpone the discussion about the architecture used to generate an image). Let us denote by $\mathcal{D}(x)$ the score assigned by the critic to an image where $\mathcal{D}(x)$ should be high if $x$ is real and low if $x$ is a fake. In the original GAN formulation, the critic $D$ was trying to solve a binary classification problem with a binary cross entropy loss. This has been shown to have various issues : training was unstable, the gradient is vanishing and not sufficient for correctly training the generator, and the generator tends to generate a single type of fooling images (mode collapse). Later, variations known as Deep Convolutional GAN (DCGAN, [@Radford2016]) and Wasserstein GAN (WGAN, [@Arjovsky2017]) were introduced and solve these problems. 
In WGAN, the critic tries to separate the scores assigned to real images from the scores assigned to fake images and it tries to **maximize** : 

$$
\mathcal{L}_c = \frac{1}{m} \sum_{i=1}^m D(x_i) - D(G(z_i))
$$

The generator tries to fool the critic by **maximiz**ing the score the critic will assign to its generated image

$$
\mathcal{L}_g = \frac{1}{m} \sum_{i=1}^m D(G(z_i))
$$

## Lab work materials

You are provided with some starter code which already implements some functionalities :


## Implementing WGAN

### Implementing the critic

### Implementing the generator

### Training


While training, you can move on the next section where you will load a pretrained network.

## Experimenting WGAN

### Generating fake images

### Interpolating in the latent space





## References
