import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pprint

# The datasets is already downloaded on the cluster
dataset_dir = "/opt/Datasets/Pascal-VOC2012/"
download = False

# How do we preprocessing the image (e.g. none, crop, shrink)
image_transform_params = {'image_mode': 'none'}

# How do we preprocess the targets
target_transform_params = {'target_mode': 'preprocessed'}

# The post-processing of the image
image_transform = transforms.ToTensor()


train_dataset, valid_dataset = data.make_trainval_dataset(
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)

print(train_dataset[0])

# Displaying an image
img, label = train_dataset[0]
img = np.transpose(img.numpy(), (1,2,0))

plt.figure()
plt.imshow(img)
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.show()


# Getting the dimensions of the images
widths = []
heights = []


#count = 0
for img, label in tqdm.tqdm(train_dataset):
    heights.append(img.shape[1])
    widths.append(img.shape[2])
    #count += 1
    #if (count >= 1000):
    #    break

sizes = np.zeros((len(widths), 2))
sizes[:,0] = heights
sizes[:,1] = widths

plt.figure()
n, bins, patches = plt.hist(sizes, 10, histtype='bar',
        color=['crimson', 'chartreuse'],
        label=['height', 'width'])
print("Bins : {}".format(bins))
plt.legend()
plt.savefig("histo_sizes.png", bbox_inches='tight')
plt.show()


# Counting the number of objects of each class
count = {k:0 for k in data.classes}
for img, label in tqdm.tqdm(train_dataset):
    for obj in label['annotation']['object']:
        count[obj['name']] += 1

pprint.pprint(count)
