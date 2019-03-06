import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import os


class DatasetTransformer(torch.utils.data.Dataset):

    """
        DatasetTransformer :

            Takes a dataset which returns a pair (inputs, targets)
            and a transform which is applied to the inputs

            Might be usefull to apply different set of transforms
            to the same dataset
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


# Hold the class names
# This is filled as soon as load_data is called once
classes = []

def load_data(valid_ratio: float,
              batch_size: int,
              num_workers: int,
              dataset_dir: str,
              train_augment_transform: list):

    if not dataset_dir:
        dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets')
        print("dataset_dir is not defined, the data will be saved into {}".format(dataset_dir))


    # Load the CIFAR-100 dataset
    train_valid_dataset = torchvision.datasets.CIFAR100(dataset_dir,
                                                        train=True,
                                                        transform=None,
                                                        target_transform=None,
                                                        download=True)

    train_valid_dataset._load_meta()
    global classes
    classes = train_valid_dataset.classes

    # Random split into traindataset and validation dataset
    nb_train, nb_valid = int((1.0 - valid_ratio) * len(train_valid_dataset)), int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

    data_transforms = {'train' : transforms.ToTensor(),
                       'valid' : transforms.ToTensor()}

    train_dataset = DatasetTransformer(train_dataset, data_transforms['train'])
    valid_dataset = DatasetTransformer(valid_dataset, data_transforms['valid'])


    # Build up the dataloaders to construct the mini batches
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=batch_size,
                                              shuffle=False, # no need to shuffle
                                              num_workers=num_workers)

    return train_loader, valid_loader



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    valid_ratio = 0.2
    batch_size  = 16
    num_workers = 2
    dataset_dir = None
    train_augment_transform = []

    train_loader, valid_loader = load_data(valid_ratio, batch_size, num_workers, dataset_dir, train_augment_transform)

    print(classes)

    # Get a minibatch
    imgs, targets = next(iter(train_loader))

    # And plot some of them
    fig, axes = plt.subplots(1,4, figsize=(8,2))
    for iax, ax in enumerate(axes):
        ax.imshow(imgs[iax].permute((1,2,0)).numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Class : {}'.format(classes[targets[iax].item()]))
    plt.show()

