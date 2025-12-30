import torch.nn as nn
import torch.nn.utils.rnn


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
            if predictions.unsorted_indices is not None:
                B = predictions.unsorted_indices.shape[0]
            elif predictions.batch_sizes is not None:
                B = predictions.batch_sizes[0]
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
