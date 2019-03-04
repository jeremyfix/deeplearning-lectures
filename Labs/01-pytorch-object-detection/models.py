import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch



class SingleBboxHead(nn.Module):

    def __init__(self, num_features: int, num_classes: int):
        super(SingleBboxHead, self).__init__()
        #####    #####
        # TO BE DONE #
        #vvvvvvvvvvvv#

        # Head bbox should output 4 numbers for cx, cy, width, height that can lie in [0, 1]
        self.head_bbox =
        # Head class should output the logits over the 20 classes
        self.head_class =

        #^^^^^^^^^^^^#
        #####    #####

    def forward(self, features):
        # We might get feature maps as input
        # that we need to "linearize"
        features = features.view(features.size()[0], -1)

        y_bbox  = self.head_bbox(features)
        y_class = self.head_class(features)

        return y_bbox, y_class

class MultipleBboxHead(nn.Module):
    def __init__(self, num_channels: int,
                 num_classes: int,
                 num_box: int):
        super(MultipleBboxHead, self).__init__()
        #####    #####
        # TO BE DONE #
        #vvvvvvvvvvvv#

        # The Bbox head outputs 4 x num_box  numbers per grid cell
        # i.e. every grid cell is predicting num_box bounding boxes
        self.head_bbox = ....

        # The class head outputs a distribution over the classes
        # and a number which is the probability that the
        # cell contains an object
        self.head_class =

        #^^^^^^^^^^^^#
        #####    #####

    def forward(self, features):

        # y_bbox is of size   Batch, 4, num_cells, num_cells
        y_bbox  = self.head_bbox(features)

        # Reminder : y_class is of size Batch, 21, num_cells, num_cells
        # 20 logits (class scores) + 1 logit (score for hosting an object)
        y_class = self.head_class(features)

        return y_bbox, y_class[:, :-1, :, :], y_class[:,-1, :, :]
