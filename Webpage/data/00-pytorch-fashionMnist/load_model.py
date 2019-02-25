import argparse
from argparse import RawTextHelpFormatter
import os
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms

import data
import models
import utils

parser = argparse.ArgumentParser(description="""
Loads and tests a model.\n

Example usage :\n

python3 load_model.py --model linear --paramfile ./logs/linear_4/best_model.pt
""", formatter_class=RawTextHelpFormatter)

parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Whether to use GPU'
        )

parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Where to store the downloaded dataset',
        default=os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')

        )

parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
        )


parser.add_argument(
        '--model',
        choices=['linear', 'fc', 'fcreg', 'vanilla', 'fancyCNN'],
        action='store',
        required=True
        )


parser.add_argument(
        '--paramfile',
        type=str,
        help='Which parameters to load, contains state_dict. Should be the full model, not only the state_dict',
        required=True
        )

args = parser.parse_args()


img_size = (1, 28, 28)
num_classes = 10
batch_size = 128
dataset_dir = args.dataset_dir
use_gpu = args.use_gpu

test_transforms = transforms.ToTensor()

test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
        train=False,
        transform=test_transforms)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=args.num_workers)



best_model_path = args.paramfile
#model = models.build_model(args.model, img_size, num_classes)
#model.load_state_dict(torch.load(best_model_path))

model = torch.load(best_model_path)
if not issubclass(type(model), torch.nn.Module):
    print("Your param file {} does not seem to be the right one. You should provide a model that has been fully saved with torch.save(model, filepath) and not only the state_dict of the model.".format(best_model_path))
    sys.exit(-1)

print(utils.torch_summarize(model))

print("{} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

model.eval()

criterion = torch.nn.CrossEntropyLoss()

t0 = time.time()
test_loss, test_acc = utils.test(model, test_loader, criterion, use_gpu)
t1 = time.time()
print("Evaluation in {} s.".format(t1 - t0))
print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))
