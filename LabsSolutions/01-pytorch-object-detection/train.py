#python3 train_one_object.py --train_dataset train.pt --valid_dataset valid.pt --num_workers 6 --use_gpu


# QUESTION :
#  1) Why do I need to load the tensor on CPU with torch.load ?
#

import argparse
import os
import sys

import torch
import torch.utils.data
import torch.nn as nn
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

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
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--tensors',
        type=str,
        help='Where to find the input tensors. We expect <tensors>/train_xx.pt and <tensors>/valid_xx.pt',
        required=True
    )

    parser.add_argument(
        '--target_mode',
        choices=['all_bbox', 'largest_bbox'],
        help='Which filter to apply to the targets',
        action='store',
        required=True
    )

    parser.add_argument(
        '--logdir',
        type=str,
        help='Where to store logs',
        default='./logs'
    )


    args = parser.parse_args()

    num_classes = len(data.classes)
    batch_size = 256
    epochs = 100


    logdir = utils.generate_unique_logpath(args.logdir, args.target_mode)
    os.makedirs(logdir)
    print("The logs will be saved in {}".format(logdir))

    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')

    #train_data = torch.load(args.train_dataset, map_location=torch.device('cpu'))
    #valid_data = torch.load(args.valid_dataset, map_location=torch.device('cpu'))



    import sys
    sys.exit(-1)


    print("The train data provides the following keys : {}".format(",".join(train_data.keys())))
    if(args.target_mode == 'largest_bbox'):
        train_dataset = torch.utils.data.TensorDataset(train_data['features'],
                                                       train_data['bboxes'],
                                                       train_data['labels'])
        valid_dataset = torch.utils.data.TensorDataset(valid_data['features'],
                                                       valid_data['bboxes'],
                                                       valid_data['labels'])
    else:
        train_dataset = torch.utils.data.TensorDataset(train_data['features'],
                                                       train_data['bboxes'],
                                                       train_data['labels'],
                                                       train_data['has_obj'])
        valid_dataset = torch.utils.data.TensorDataset(valid_data['features'],
                                                       valid_data['bboxes'],
                                                       valid_data['labels'],
                                                       valid_data['has_obj'])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory = True,
            num_workers=args.num_workers)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory = True,
            num_workers=args.num_workers)


    one_sample = train_dataset[0] # features, bbox, label
    num_features = one_sample[0].numel()
    num_channels = one_sample[0].size()[0]

    ############################################ Model
    if(args.target_mode == 'largest_bbox'):
        model = models.SingleBboxHead(num_features, num_classes)
    else:
        model = models.MultipleBboxHead(num_channels, num_classes, 1)
    model = model.to(device=device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    ############################################ Train


    tensorboard_writer   = SummaryWriter(log_dir = logdir)
    model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt", model)

    num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary_file = open(logdir + "/summary.txt", 'w')
    summary_text = """

Executed command
================
{}

Dataset
=======
PascalVOC preprocessed

Model summary
=============
    {}

{} trainable parameters

Optimizer
========
    {}
    """.format(" ".join(sys.argv),
               str(model).replace('\n','\n\t'),
               num_params,
               str(optimizer).replace('\n','\n\t'))
    summary_file.write(summary_text)
    summary_file.close()

    tensorboard_writer.add_text("Experiment summary", summary_text)
    print("{} parameters to be optimized".format(num_params))




    if(args.target_mode == 'largest_bbox'):
        for t in range(epochs):
            print("Epoch {}".format(t))
            scheduler.step()

            train_reg_loss, train_acc = utils.train(model, train_loader, optimizer, device)

            val_reg_loss, val_acc = utils.test(model, valid_loader, device)
            print(" Validation : Bbox Loss : {:.4f}, Class Acc : {:.4f}".format(val_loss, val_acc))

            tensorboard_writer.add_scalar('metrics/train/bbox_reg_loss', train_loss, t)
            tensorboard_writer.add_scalar('metrics/train/class_acc',  train_acc, t)
            tensorboard_writer.add_scalar('metrics/val/bbox_reg_loss', val_loss, t)
            tensorboard_writer.add_scalar('metrics/val/class_acc',  val_acc, t)
            model_checkpoint.update(val_reg_loss + (1.0 - val_acc))
    else:
        for t in range(epochs):

            scheduler.step()

            print("Epoch {}".format(t))
            train_reg_loss, train_acc, train_obj = utils.train_multiple_objects(model, train_loader, optimizer, device)

            val_reg_loss, val_acc, val_obj = utils.test_multiple_objects(model, valid_loader, device)
            print(" Validation : Bbox Loss : {:.4f}, Class Acc : {:.4f}, Obj loss: {:.4f}".format(val_reg_loss, val_acc, val_obj))

            tensorboard_writer.add_scalar('metrics/train/bbox_reg_loss', train_reg_loss, t)
            tensorboard_writer.add_scalar('metrics/train/class_acc',  train_acc, t)
            tensorboard_writer.add_scalar('metrics/train/has_obj',  train_obj, t)
            tensorboard_writer.add_scalar('metrics/val/bbox_reg_loss', val_reg_loss, t)
            tensorboard_writer.add_scalar('metrics/val/class_acc',  val_acc, t)
            tensorboard_writer.add_scalar('metrics/val/has_obj',  val_obj, t)
            model_checkpoint.update(val_reg_loss + val_obj + (1.0 - val_acc))




