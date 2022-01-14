#!/usr/bin/env python3

# External modules
import torchvision.transforms as transforms
import torch
# Local modules
import data

objects = [{'bndbox': {'cx': 0.524, 'cy': 0.5735294117647058, 'width': 0.836, 'height': 0.753393665158371}, 'class': 5},
                   {'bndbox': {'cx': 0.447, 'cy': 0.23868778280542988, 'width': 0.262, 'height': 0.27828054298642535}, 'class': 0}]


largest_object = data.filter_largest(objects)
# Expected: {'bndbox': {'cx': 0.524, 'cy': 0.5735294117647058, 'width': 0.836, 'height': 0.753393665158371}, 'class': 5}
print(largest_object)

target_tensor = data.target_to_tensor(largest_object)
# Expected : {'bboxes': torch.tensor([0.5240, 0.5735, 0.8360, 0.7534]), 'labels': torch.tensor([5])})
print(target_tensor)

# The datasets is already downloaded on the cluster
dataset_dir = "/mounts/Datasets2/Pascal-VOC2012/"
download = False

# How do we preprocessing the image (e.g. none, crop, shrink)
image_transform_params = {'image_mode': 'none'}

# How do we preprocess the targets
target_transform_params = {'target_mode': 'largest_bbox', 'image_transform_params': image_transform_params}

# The post-processing of the image
image_transform = transforms.ToTensor()


train_dataset, valid_dataset = data.make_trainval_dataset(
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)

print(train_dataset[203])


