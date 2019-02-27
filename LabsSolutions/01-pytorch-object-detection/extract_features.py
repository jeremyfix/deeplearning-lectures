
import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms


import data
import models
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Whether to use GPU'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to enter debug mode; For example limiting the number of extracted samples. It extracts the features of a single batch'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Where to store the downloaded dataset',
        default=None
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Whether to download the dataset'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--image_mode',
        choices=['none', 'shrink', 'crop'],
        default='shrink',
        help='Which method to apply to resize the input images',
        action='store'
    )

    parser.add_argument(
        '--target_mode',
        choices=['all_bbox', 'largest_bbox'],
        help='Which filter to apply to the targets',
        action='store',
        required=True
    )


    parser.add_argument(
        '--model',
        choices=['resnet18', 'resnet34','resnet50','resnet152','densenet121','squeezenet1_1'],
        help='Which pretrained model to use to compute the features',
        action='store',
        required=True
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./tensors',
        help='Where to store the precomputed tensors'
    )

    args = parser.parse_args()

    output_image_size = {'width': 224, 'height': 224}
    image_transform_params = {'image_mode': args.image_mode, 'output_image_size': output_image_size}
    num_cells = 7 # Used only if necessary , i.e. target_mode is all_bbox
    target_transform_params = {'target_mode': args.target_mode, 'image_transform_params': image_transform_params, 'num_cells': num_cells}

    batch_size=128

    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    ############################################ Datasets and Dataloaders

    model_preprocessing = models.preprocessings[args.model]

    image_transform = transforms.Compose([transforms.ToTensor(), model_preprocessing])

    train_dataset, valid_dataset = data.make_trainval_dataset(dataset_dir  = args.dataset_dir,
                                                              image_transform_params = image_transform_params,
                                                              transform    = image_transform,
                                                              target_transform_params  = target_transform_params,
                                                              download     = args.download)

    print("{} training samples have been loaded\n{} validation samples have been loaded".format(len(train_dataset), len(valid_dataset)))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
            batch_size=batch_size,
            shuffle = False,
            pin_memory = args.use_gpu,
            num_workers=args.num_workers)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory = args.use_gpu,
            num_workers=args.num_workers)


    ############################################ Model
    model = models.FeatureExtractor(model_name = args.model)
    model = model.to(device=device)

    ########################################### Preprocessing the dataset to extract the features
    print("Extracting and saving the features for the training set")
    utils.extract_save_features(train_loader, model, device, args.output_dir + "/train_")

    print("Extracting and saving the features for the validation set")
    utils.extract_save_features(valid_loader, model, device, args.output_dir + "/valid_")

    print("Done.")



