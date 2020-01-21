#!/usr/bin/env python3

# Standard modules
import os.path
import copy
# External modules
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomAffine
import numpy as np


def get_data_loaders(config):
    if config['data_augment']:
        train_augment_transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                       RandomAffine(degrees=10, translate=(0.1, 0.1))])
    else:
        train_augment_transforms = None
    batch_size = config['batch_size']
    valid_ratio = config['valid_ratio']
    train_loader, valid_loader, test_loader, normalization_function = load_fashion_mnist(valid_ratio,
                                                                                              batch_size,
                                                                                              config['num_workers'],
                                                                                              config['normalize'],
                                                                                              dataset_dir =
                                                                                              config['dataset_dir'],
                                                                                              train_augment_transforms = train_augment_transforms)
    return train_loader, valid_loader, test_loader, normalization_function, train_augment_transforms


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

class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
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
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return test_loader


class CenterReduce:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x-self.mean)/self.std

def load_fashion_mnist(valid_ratio, batch_size,
        num_workers, normalize,
        dataset_dir=None, train_augment_transforms=None,
        normalizing_tensor_path=None):

    if not dataset_dir:
        dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')


    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                               train=True,
                                               download=True)

    # Split it into training and validation sets
    nb_train, nb_valid = int((1.0 - valid_ratio) * len(train_valid_dataset)), int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])


    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                     train=False)

    # Do we want to normalize the dataset given the statistics of the training set ?
    data_transforms = {'train' : transforms.ToTensor(),
                       'valid' : transforms.ToTensor(),
                       'test'  : transforms.ToTensor() }

    if train_augment_transforms:
        data_transforms['train'] = transforms.Compose([train_augment_transforms, transforms.ToTensor()])

    if normalize:

        normalizing_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
        normalizing_loader = torch.utils.data.DataLoader(dataset=normalizing_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers)

        # Compute mean and variance from the training set
        mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)


        normalization_function = CenterReduce(mean_train_tensor,
                                              std_train_tensor)

        # Apply the transformation to our dataset
        for k, old_transforms in data_transforms.items():
            data_transforms[k] = transforms.Compose([
                old_transforms,
                transforms.Lambda(lambda x: normalization_function(x))
                ])
    else:
        normalization_function = None

    train_dataset = DatasetTransformer(train_dataset, data_transforms['train'])
    valid_dataset = DatasetTransformer(valid_dataset, data_transforms['valid'])
    test_dataset  = DatasetTransformer(test_dataset , data_transforms['test'])

    # shuffle = True : reshuffles the data at every epoch
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)


    return train_loader, valid_loader, test_loader, copy.copy(normalization_function)


def display_tensor_samples(tensor_samples, labels=None, filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm


    fig=plt.figure(figsize=(20,5),facecolor='w')
    nsamples= tensor_samples.shape[0]
    for i in range(nsamples):
        ax = plt.subplot(1,nsamples, i+1)
        plt.imshow(tensor_samples[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        #plt.axis('off')
        if labels is not None:
            ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def display_samples(loader, nsamples, filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    imgs, labels = next(iter(train_loader))
    display_tensor_samples(imgs[:nsamples], labels[:nsamples], filename)
    #print("imgs is of shape {},  labels of shape {}'".format(imgs.shape, labels.shape))

    #fig=plt.figure(figsize=(20,5),facecolor='w')
    #for i in range(nsamples):
    #    ax = plt.subplot(1,nsamples, i+1)
    #    plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
    #    #plt.axis('off')
    #    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
    #    ax.get_xaxis().set_visible(False)
    #    ax.get_yaxis().set_visible(False)
    #if filename:
    #    plt.savefig(filename, bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':

    num_threads = 4
    valid_ratio = 0.2
    batch_size = 128
    classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_loader, valid_loader, test_loader = load_fashion_mnist(valid_ratio, batch_size, num_threads, False)

    print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))
    print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

    #display_samples(train_loader, 10, 'fashionMNIST_samples.png')

    ###################################################################################
    ## Data augmentation

    train_valid_dataset = torchvision.datasets.FashionMNIST(root=os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST'),
                                               train=True,
                                               download=True)



    train_augment = transforms.Compose([transforms.RandomHorizontalFlip(0.5), RandomAffine(degrees=10, translate=(0.1,0.1))])


    # data augment a single sample several times
    img, label = train_valid_dataset[np.random.randint(len(train_valid_dataset))]
    Timg = transforms.functional.to_tensor(img)
    n_augmented_samples = 10
    aug_imgs = torch.zeros(n_augmented_samples, Timg.shape[0], Timg.shape[1],Timg.shape[2])
    for i in range(n_augmented_samples):
        aug_imgs[i] = transforms.ToTensor()(train_augment(img))
    print("I augmented a {}".format(classes_names[label]))
    display_tensor_samples(aug_imgs, filename="fashionMNIST_sample_augment.png")



    # Test with data augmentation
    train_augment = transforms.Compose([transforms.RandomHorizontalFlip(0.5), RandomAffine(degrees=10, translate=(0.1,0.1))])

    train_loader, _, _ = load_fashion_mnist(valid_ratio, batch_size, num_threads, False, train_augment_transforms=train_augment)
    display_samples(train_loader, 10, 'fashionMNIST_samples_augment.png')


    # Loading normalized datasets
    train_loader, valid_loader, test_loader = load_fashion_mnist(valid_ratio, batch_size, num_threads, True)

