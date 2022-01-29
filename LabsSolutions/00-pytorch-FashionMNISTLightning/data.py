#!/usr/bin/env python3

# Standard imports
import os.path
import copy
from typing import List
# External imports
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomAffine
import numpy as np

def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1

    return mean_img, std_img

class DatasetTransformer(Dataset):

    def __init__(self, base_dataset, transform):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


def load_test_data(batch_size, dataset_dir=None):
    if not dataset_dir:
        dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')

    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                     train=False)


    test_dataset  = DatasetTransformer(test_dataset , transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return test_loader


class CenterReduce:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x-self.mean)/self.std


def make_dataloaders(valid_ratio,
                     batch_size,
                     num_workers,
                     normalize,
                     dataaugment_train: bool = False,
                     dataset_dir=None,
                     normalizing_tensor_path=None):

    if not dataset_dir:
        dataset_dir = os.path.join(os.path.expanduser("~"),
                                   'Datasets', 'FashionMNIST')


    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                            train=True,
                                                            download=True)
    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                     train=False)

    # Split it into training and validation sets
    nb_train, nb_valid = int((1.0 - valid_ratio) * len(train_valid_dataset)), int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = random_split(train_valid_dataset,
                                                [nb_train, nb_valid])

    # Base data transform pipeline, modified latter dependening on the options
    data_transforms = {'train' : transforms.ToTensor(),
                       'valid' : transforms.ToTensor(),
                       'test'  : transforms.ToTensor() }

    if normalize:
        normalizing_dataset = DatasetTransformer(train_dataset,
                                                 transforms.ToTensor())
        normalizing_loader = DataLoader(dataset=normalizing_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers)

        # Compute mean and variance from the training set
        mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)

        normalization_function = CenterReduce(mean_train_tensor,
                                              std_train_tensor)

        # Apply the transformation to all the three datasets
        for k, old_transforms in data_transforms.items():
            data_transforms[k] = transforms.Compose([
                old_transforms,
                transforms.Lambda(lambda x: normalization_function(x))
                ])
    else:
        normalization_function = None

    if dataaugment_train:
        # Data augmentation on the training set
        # Note the default fill value for the RandomAffine is 0.0 (black)
        train_augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ])
        # Add the augmentation in the pipeline of the training set
        # transform
        data_transforms['train'] = transforms.Compose([
            train_augment_transforms,
            data_transforms['train']
        ])

    # Build the three datasets with their transforms
    train_dataset = DatasetTransformer(train_dataset,
                                       data_transforms['train'])
    valid_dataset = DatasetTransformer(valid_dataset,
                                       data_transforms['valid'])
    test_dataset  = DatasetTransformer(test_dataset,
                                       data_transforms['test'])

    # Build the three dataloaders with shuffling at every epoch
    # for the training set
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)


    test_loader =  DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers)


    return (train_loader, valid_loader, test_loader), copy.copy(normalization_function)


def display_tensor_samples(tensor_samples,
                           labels,
                           filename,
                           classes_names: List[str]):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig=plt.figure(figsize=(20,3),facecolor='w')
    nsamples= tensor_samples.shape[0]

    for i in range(nsamples):
        ax = plt.subplot(1,nsamples, i+1)
        plt.imshow(tensor_samples[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        #plt.axis('off')
        if isinstance(labels, torch.Tensor):
            ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if not isinstance(labels, torch.Tensor):
        # Consider labels as a super title
        plt.suptitle(classes_names[labels])

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def display_samples(loader, nsamples,
                    filename,
                    classes_names):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    imgs, labels = next(iter(loader))
    display_tensor_samples(imgs[:nsamples],
                           labels[:nsamples],
                           filename,
                           classes_names)


if __name__ == '__main__':

    num_threads = 4
    valid_ratio = 0.2
    batch_size = 128
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover',
                     'Dress', 'Coat', 'Sandal','Shirt',
                     'Sneaker', 'Bag', 'Ankle boot']

    ###################################################################################
    # Simple data loader creation
    loaders, fnorm = make_dataloaders(valid_ratio,
                                      batch_size,
                                      num_threads,
                                      False,
                                      False,
                                      None,
                                      None)
    train_loader, valid_loader, test_loader = loaders
    print(f"The train set contains {len(train_loader.dataset)} images, in {len(train_loader)} batches")
    print(f"The validation set contains {len(valid_loader.dataset)} images, in {len(valid_loader)} batches")
    print(f"The test set contains {len(test_loader.dataset)} images, in {len(test_loader)} batches")

    display_samples(train_loader, 10, 'fashionMNIST_samples.png', classes_names)

    ###################################################################################
    ## Data augmentation

    loaders, fnorm = make_dataloaders(valid_ratio,
                                      batch_size,
                                      num_threads,
                                      False,
                                      True,
                                      None,
                                      None)
    # Let us take the first sample of the dataset and sample it several
    # times 
    train_loader, _, _ = loaders
    sample_idx = 0
    samples = [train_loader.dataset[sample_idx][0] for i in range(10)]
    label = train_loader.dataset[sample_idx][1] 

    # Build a torch tensor from the list of samples
    samples = torch.cat(samples, dim=0).unsqueeze(dim=1) # to add C=1

    display_tensor_samples(samples, label,
                           'fashionMNIST_sample_aug.png',
                           classes_names)

