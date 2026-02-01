# coding: utf-8

# External imports
import torch

# Local imports
from nirlab.optim import RelativeMSE, KSpaceLoss


class GenericBatchMetric:
    def __init__(self, metric):
        """
        Args:
            metric: a batch averaged metric to accumulate
        """
        self.metric = metric
        self.cum_metric = 0.0
        self.num_samples = 0

    def reset(self):
        self.cum_metric = 0
        self.num_samples = 0

    def __call__(self, predictions, targets):
        """
        predictions: (B, *)
        targets : (B, *)
        """
        # We suppose is batch averaged
        if isinstance(predictions, torch.nn.utils.rnn.PackedSequence):
            B = predictions.unsorted_indices.shape[0]
        else:
            B = predictions.shape[0]
        self.cum_metric += B * self.metric(predictions, targets).item()
        self.num_samples += B

    def get_value(self):
        if self.num_samples == 0:
            raise ZeroDivisionError
        return self.cum_metric / self.num_samples

    def __str__(self):
        return f"{self.get_value():.3f}"

    def tensorboard_write(self, writer, prefix, global_step):
        writer.add_scalar(prefix, self.get_value(), global_step)


def BatchRelativeMSE():
    return GenericBatchMetric(RelativeMSE())

class KSpaceMetric:
    def __init__(self):
        self.loss = KSpaceLoss()
        self.value = None

    def reset(self):
        pass

    def __call__(self, predictions, targets):
        self.value = self.loss(predictions, targets)
    
    def get_value(self):
        return self.value.item()
    
    def __str__(self):
        return f"{self.get_value():.3f}"
    
    def tensorboard_write(self, writer, prefix, global_step):
        writer.add_scalar(prefix, self.get_value(), global_step)