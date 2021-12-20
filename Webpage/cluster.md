---
title:  'Using the GPU cluster of CentraleSupelec'
author:
- Jeremy Fix
keywords: [CentraleSupelec, GPUs]
...


To connect to the GPUs at CentraleSupelec, I strongly recommend you to use [dcejs](https://github.com/jeremyfix/dcejs), a cross platform desktop application which allows to easily book a node and start a graphical session on the allocated node as illustrated below

## Connecting to the GPUs during a lab session

During a lab session, you are provided with a reservation code that you must use for your labwork. The video below shows how to perform an allocation and start a graphical session with an example reservation called **demoresa**.

<video width="50%" controls autoplay>
  <source src="./data/gpu_resa.mp4" type="video/mp4">
</video>

## Connecting to the GPUs during free times to finish a labwork

You have the possibility to allocate a GPU for working after the official lab work sessions. In that case, you must allocate a node on the partition **gpu_tp**. Every session cannot last more than 2 hours. If you want to work for more than 2 hours in a row, you will have to make a new allocation once the first one ends.

Below, I show you how to allocate a node on **gpu_tp** for 2 hours.

<video width="50%" controls autoplay>
  <source src="./data/tp_dce.mp4" type="video/mp4">
</video>

