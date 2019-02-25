import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pprint

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
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)

img, target = train_dataset[203]
print(type(img))
img.save('bird.jpeg')
#orig_img = np.transpose(train_dataset[img_idx][0].numpy(), (1,2,0))

print(orig_img.shape)

image_transform_params = {'image_mode': 'shrink', 'output_image_size': {'width':224, 'height': 224}}
train_dataset, valid_dataset = data.make_trainval_dataset(
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)
shrink_img = np.transpose(train_dataset[img_idx][0].numpy(), (1,2,0))


image_transform_params = {'image_mode': 'crop', 'output_image_size':{'width':224, 'height': 224}}
train_dataset, valid_dataset = data.make_trainval_dataset(
        dataset_dir             = dataset_dir,
        image_transform_params  = image_transform_params,
        transform               = image_transform,
        target_transform_params = target_transform_params,
        download                = download)
crop_img =np.transpose(train_dataset[img_idx][0].numpy(), (1,2,0))



# Displaying an image
img, label = train_dataset[0]
img = np.transpose(img.numpy(), (1,2,0))

fig = plt.figure(figsize=(15,5))
axes = fig.subplots(1,3)
axes[0].imshow(orig_img, aspect='equal')
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


