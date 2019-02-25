import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss

import argparse

import models
import utils
import data

####################################################### Arguments
parser = argparse.ArgumentParser()


parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Whether to use GPU'
)


args = parser.parse_args()

if args.use_gpu:
    print("Using GPU{}".format(torch.cuda.current_device()))
    device = torch.device('gpu')
else:
    print("Using CPU")
    device = torch.device('cpu')


params_paths = ["./logs/linear_3", "./logs/linear_4",
        "./logs/linear_5"]

batch_size = 256
num_classes = 10

# Our average model is outputing probabilities
# we need to define a a loss combining NLLLoss and Log applied
# to the softmax of our averaged model
class CELoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction='mean'):
        super(CELoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.nll_loss(torch.log(input), target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)



class AverageModel:

    def __init__(self, params_paths, device, num_classes):
        self.num_classes = num_classes
        self.normalization_functions = []
        self.models = []
        self.device = device

        for p in params_paths:
            model_path = os.path.join(p, "best_model.pt")
            loaded_dict  = torch.load(model_path)

            model = loaded_dict['model'].to(device)

            self.models.append(model)
            self.normalization_functions.append(loaded_dict['normalization_function'])


    def eval(self):
        for m in self.models:
            m.eval()

    def __call__(self, x):
        mean_prob = torch.zeros(x.shape[0], self.num_classes)
        mean_prob = mean_prob.to(device)

        for m,n in zip(self.models, self.normalization_functions):
            if n:
                x = n(x)
            y = nn.functional.softmax(m(x), 1)
            mean_prob += y / len(self.models)

        return mean_prob


test_loader = data.load_test_data(batch_size)

loss = nn.CrossEntropyLoss()
avg_model = AverageModel(params_paths, device, num_classes)

print("Individual performances : ")
for i, m in enumerate(avg_model.models):
    test_loss, test_acc = utils.test(m, test_loader, loss, device)
    print("[{}]    Test : Loss : {:.4f}, Acc : {:.4f}".format(params_paths[i], test_loss, test_acc))


print("="*30)
print("Averaged model : ")
loss = CELoss()
test_loss, test_acc = utils.test(avg_model, test_loader, loss, device)
print("Test : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))

