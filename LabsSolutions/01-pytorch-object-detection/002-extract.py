import sol_data as data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import sys
import utils

device = torch.device('cpu')
batch_size = 128
num_workers = 7

# The datasets is already downloaded on the cluster
dataset_dir = "/mounts/Datasets2/Pascal-VOC2012/"
download = False

# How do we preprocessing the image (e.g. none, crop, shrink)
image_transform_params = {'image_mode': 'shrink', 'output_image_size': {'width':224, 'height':224}}

# How do we preprocess the targets
target_transform_params = {'target_mode': 'largest_bbox', 'image_transform_params': image_transform_params}

# The post-processing of the image
image_transform = transforms.ToTensor()

imagenet_preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

image_transform = transforms.Compose([transforms.ToTensor(), imagenet_preprocessing])

train_dataset, valid_dataset = data.make_trainval_dataset(
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        shuffle = False,
        num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)


model = torchvision.models.resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(model.children())[:-2])
feature_extractor.to(device)

utils.extract_save_features(train_loader, feature_extractor, device, 'train.pt')
utils.extract_save_features(valid_loader, feature_extractor, device, 'valid.pt')
