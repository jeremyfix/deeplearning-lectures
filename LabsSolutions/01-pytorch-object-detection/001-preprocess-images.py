#!/usr/bin/env python3

# External modules
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pprint
# Local modules
import data

# The datasets is already downloaded on the cluster
dataset_dir = "/opt/Datasets/Pascal-VOC2012/"
download = False

# How do we preprocess the targets
target_transform_params = {'target_mode': 'orig'}

# The post-processing of the image
image_transform = None#transforms.ToTensor()

img_idx = 203

# How do we preprocessing the image (e.g. none, crop, shrink)
image_transform_params = {'image_mode': 'none'}
train_dataset, valid_dataset = data.make_trainval_dataset(
    dataset_dir=dataset_dir,
    image_transform_params=image_transform_params,
    transform=image_transform,
    target_transform_params=target_transform_params,
    download=download)

img, target = train_dataset[img_idx]
print("The image from the dataset is of type {}".format(type(img)))

print("Saving an image as bird.jpeg")
img.save('bird.jpeg')

img = np.asarray(img)
print(img.shape)

image_transform_params = {'image_mode': 'shrink',
                          'output_image_size': {'width': 224, 'height': 224}}
train_dataset, valid_dataset = data.make_trainval_dataset(
    dataset_dir=dataset_dir,
    image_transform_params=image_transform_params,
    transform=image_transform,
    target_transform_params=target_transform_params,
    download=download)
shrink_img = np.asarray(train_dataset[img_idx][0])

image_transform_params = {'image_mode': 'crop',
                          'output_image_size': {'width': 224, 'height': 224}}
train_dataset, valid_dataset = data.make_trainval_dataset(
    dataset_dir=dataset_dir,
    image_transform_params=image_transform_params,
    transform=image_transform,
    target_transform_params=target_transform_params,
    download=download)
crop_img = np.asarray(train_dataset[img_idx][0])


# Displaying an image
fig = plt.figure(figsize=(15, 5))
axes = fig.subplots(1, 3)
axes[0].imshow(img, aspect='equal')
axes[0].get_xaxis().set_visible(False)
axes[0].get_yaxis().set_visible(False)
axes[0].set_title('Original image')

axes[1].imshow(shrink_img, aspect='equal')
axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)
axes[1].set_title('Shrink')

axes[2].imshow(crop_img, aspect='equal')
axes[2].get_xaxis().set_visible(False)
axes[2].get_yaxis().set_visible(False)
axes[2].set_title('Crop')

plt.savefig('preprocess_images.png', bbox_inches='tight')
plt.show()


