import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss

import models
import utils
import data

use_gpu = False #True
params_paths = ["./logs/fancyCNN_38", "./logs/fancyCNN_39", "./logs/fancyCNN_40", "./logs/fancyCNN_41", "./logs/fancyCNN_42", "./logs/fancyCNN_43"]

batch_size = 10000
num_classes = 10

class CELoss(torch.nn.modules.loss._WeightedLoss):


    def __init__(self, weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction='elementwise_mean'):
        super(CELoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.nll_loss(torch.log(input), target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)



class AverageModel:

    def __init__(self, params_paths, use_gpu, num_classes):
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.models = []

        for p in params_paths:
            model_path = os.path.join(p, "best_model.ptfull")
            model = torch.load(model_path)
            if use_gpu:
                model.cuda()

            self.models.append(model)


    def eval(self):
        for m in self.models:
            m.eval()

    def __call__(self, x):
        mean_prob = torch.zeros(x.shape[0], self.num_classes)
        if self.use_gpu:
            mean_prob = mean_prob.cuda()
        for m in self.models:
            y = nn.functional.softmax(m(x), 1)
            mean_prob += y / len(self.models)

        return mean_prob


test_loader = data.load_test_data(batch_size)

loss = nn.CrossEntropyLoss()
avg_model = AverageModel(params_paths, use_gpu, num_classes)

print("Individual performances : ")
for i, m in enumerate(avg_model.models):
    test_loss, test_acc = utils.test(m, test_loader, loss, use_gpu)
    print("[{}]    Test : Loss : {:.4f}, Acc : {:.4f}".format(params_paths[i], test_loss, test_acc))


print("="*30)
print("Averaged model : ")
loss = CELoss()
test_loss, test_acc = utils.test(avg_model, test_loader, loss, use_gpu)
print("Test : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))

