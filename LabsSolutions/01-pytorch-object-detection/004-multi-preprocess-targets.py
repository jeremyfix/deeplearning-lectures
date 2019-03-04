import sol_data as data
import torchvision.transforms as transforms
import torch

objects = [{'bndbox': {'cx': 0.60, 'cy': 0.65, 'width': 0.84, 'height': 0.75}, 'class': 5},
           {'bndbox': {'cx': 0.40, 'cy': 0.20, 'width': 0.26, 'height': 0.27}, 'class': 0}]


output = data.targets_to_grid_cell_tensor(objects, 4)

print("bboxes : {}".format(output['bboxes']))
print("has_obj : {}".format(output['has_obj']))
print("labels : {}".format(output['labels']))
