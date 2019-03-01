import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.modules.module import _addindent
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict
from tqdm import tqdm

def extract_save_features(loader: torch.utils.data.DataLoader,
                          model: torch.nn.Module,
                          device: torch.device,
                          filename_prefix: str):

    with torch.no_grad():
        model.eval()
        batch_idx = 1
        for (inputs, targets) in tqdm(loader):

            inputs = inputs.to(device=device)

            # Compute the forward propagation through the body
            # just to extract the features
            features = model(inputs)

            torch.save(dict([("features", features)] + [(k, v.squeeze()) for (k,v) in targets.items()]),
                       filename_prefix+"{}.pt".format(batch_idx))

            batch_idx += 1


def train(model: torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          device: torch.device):
    """
        Train a model for one epoch, iterating over the loader
        using the f_loss to compute the loss and the optimizer
        to update the parameters of the model.

        Arguments :
        model      -- A torch.nn.Module object
        loader     -- A torch.utils.data.DataLoader
        optimizer  -- A torch.optim.Optimzer object
        device     -- The device to use for the computation CPU or GPU

        Returns :

    """


    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...

    bbox_loss  = torch.nn.SmoothL1Loss()
    bbox_reg_loss = torch.nn.L1Loss(reduction='sum')
    class_loss = torch.nn.CrossEntropyLoss()

    alpha_bbox = 20.0

    model.train()
    N = 0
    regression_loss, correct = 0.0, 0
    for i, (inputs, bboxes, labels) in enumerate(loader):

        inputs, bboxes, labels = inputs.to(device=device), bboxes.to(device=device), labels.to(device=device)

        # Compute the forward propagation
        outputs = model(inputs)

        # On the first steps, b_loss around 0.1, c_loss around 20.0
        b_loss = alpha_bbox * bbox_loss(outputs[0], bboxes)
        c_loss = class_loss(outputs[1], labels)


        # Accumulate the number of processed samples
        N += inputs.shape[0]

        # For the total loss
        regression_loss += bbox_reg_loss(outputs[0], bboxes).item()/4.0

        # For the total accuracy
        predicted_targets = outputs[1].argmax(dim=1)
        correct += (predicted_targets == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        b_loss.backward()
        c_loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass
        optimizer.step()

        # Display status
        progress_bar(i, len(loader), msg = "bbox loss : {:.4f}, classification Acc : {:.4f}".format(regression_loss/N, correct/N))
    return regression_loss/N, correct/N



def test(model, loader, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- device to be used for the computation (CPU or GPU)

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    bbox_reg_loss = torch.nn.L1Loss(reduction='sum')
    class_loss = torch.nn.CrossEntropyLoss(reduction='sum')

    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        regression_loss, correct = 0.0, 0
        for i, (inputs, bboxes, labels) in enumerate(loader):

            inputs, bboxes, labels = inputs.to(device=device), bboxes.to(device=device), labels.to(device=device)

            outputs = model(inputs)

            b_loss = bbox_reg_loss(outputs[0], bboxes)/4.0

            N += inputs.shape[0]

            # For the total bbox loss
            regression_loss += b_loss.item()

            # For the accuracy
            predicted_targets = outputs[1].argmax(dim=1)
            correct += (predicted_targets == labels).sum().item()
        return regression_loss/N, correct/N


def train_multiple_objects(model: torch.nn.Module,
          loader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          device: torch.device):
    """
        Train a model for one epoch, iterating over the loader
        using the f_loss to compute the loss and the optimizer
        to update the parameters of the model.

        Arguments :
        model      -- A torch.nn.Module object
        loader     -- A torch.utils.data.DataLoader
        optimizer  -- A torch.optim.Optimzer object
        device     -- The device to use for the computation CPU or GPU

        Returns :

    """

    alpha_bbox = 200.0

    model.train()
    Nobjects = 0
    Nsamples = 0
    regression_loss, correct, objectness_loss = 0.0, 0, 0.0
    for i, (inputs, bboxes, labels, has_obj) in enumerate(loader):

        inputs, bboxes, labels, has_obj = inputs.to(device=device), bboxes.to(device=device), labels.to(device=device), has_obj.to(device=device)
        has_obj_idx = has_obj.byte()

        # Compute the forward propagation
        outputs = model(inputs)

        predicted_bboxes = outputs[0]
        predicted_logits = outputs[1]
        predicted_hasobj = outputs[2]

        # Computes the total number of ground truth objects in the whole minibatch
        num_gt_objects = has_obj.sum().item()

        # Regression loss only where there is an object within the cell
        # otherwise, we do not care about making mistakes in the regression

        # Computes the L1 errors on all the components of the bounding box
        # difference_bbos is a (batch_size, num_cells, num_cells) tensor
        grid_cells_regression_loss = torch.sum(torch.abs(predicted_bboxes-bboxes), 1) / 4.0
        # Keep in the loss only the grid cells where an object lies
        b_loss = alpha_bbox * torch.sum(grid_cells_regression_loss[has_obj_idx]) / num_gt_objects

        # Classification loss
        numclass = predicted_logits.shape[1]

        reshaped_predicted_logits = predicted_logits.permute(0,2,3,1).contiguous().view(-1, numclass)
        reshaped_labels           = labels.view(-1).long()
        grid_cells_classification_loss = F.cross_entropy(reshaped_predicted_logits, reshaped_labels, reduction='none')
        c_loss = torch.sum(grid_cells_classification_loss[has_obj_idx.view(-1)]) / num_gt_objects

        # Objectness loss
        obj_loss = F.binary_cross_entropy(torch.sigmoid(predicted_hasobj), has_obj.float())

        # For the total loss
        regression_loss += num_gt_objects * b_loss.item()/alpha_bbox
        correct += (reshaped_predicted_logits.argmax(dim=1) == reshaped_labels)[has_obj_idx.view(-1)].sum().item()
        objectness_loss += inputs.shape[0] * obj_loss.item()

        # Accumulate the number of processed objects
        Nobjects += num_gt_objects
        Nsamples += inputs.shape[0]

        # For the total accuracy
        #predicted_targets = outputs[1].argmax(dim=1)
        #correct += (predicted_targets == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        b_loss.backward()
        c_loss.backward(retain_graph=True)
        obj_loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass
        optimizer.step()

        # Display status
        progress_bar(i, len(loader), msg = "bbox loss : {:.4f}, classification Acc : {:.4f}, obj loss: {:.4f}".format(regression_loss/Nobjects, correct/Nobjects, objectness_loss/Nsamples))
    return regression_loss/Nobjects, correct/Nobjects, objectness_loss/Nsamples



def test_multiple_objects(model, loader, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        device    -- device to be used for the computation (CPU or GPU)

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        Nobjects = 0
        Nsamples = 0
        regression_loss, correct, objectness_loss = 0.0, 0, 0.0
        for i, (inputs, bboxes, labels, has_obj) in enumerate(loader):

            inputs, bboxes, labels, has_obj = inputs.to(device=device), bboxes.to(device=device), labels.to(device=device), has_obj.to(device=device)
            has_obj_idx = has_obj.byte()

            # Computes the number of ground truth objects in the whole minibatch
            num_gt_objects = has_obj.sum().item()
            Nobjects += num_gt_objects
            Nsamples += inputs.shape[0]

            # Forward propagate through the network
            outputs = model(inputs)

            predicted_bboxes = outputs[0]
            predicted_logits = outputs[1]
            predicted_hasobj = outputs[2]

            # Regression loss
            grid_cells_regression_loss = torch.sum(torch.abs(predicted_bboxes-bboxes), 1) / 4.0
            b_loss = torch.sum(grid_cells_regression_loss[has_obj_idx])

            # Classification loss
            numclass = predicted_logits.shape[1]

            reshaped_predicted_logits = predicted_logits.permute(0,2,3,1).contiguous().view(-1, numclass)
            reshaped_labels           = labels.view(-1).long()
            correct += (reshaped_predicted_logits.argmax(dim=1) == reshaped_labels)[has_obj_idx.view(-1)].sum().item()

            # Objectness loss
            objectness_loss = F.binary_cross_entropy(torch.sigmoid(predicted_hasobj), has_obj.float(), reduction='sum')


            # For the total bbox loss
            regression_loss += b_loss.item()

        return regression_loss/Nobjects, correct/Nobjects, objectness_loss/Nsamples




# from https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py

import os
import sys
import time
import math

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [{}>{}]'.format('='*cur_len, '.'*rest_len))

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %10s' % format_time(step_time))
    L.append(' | Tot: %10s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model, self.filepath)
            self.min_loss = loss

# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    tmpstr += '\n {} learnable parameters'.format(total_params)
    return tmpstr



