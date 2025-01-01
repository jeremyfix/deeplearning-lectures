# coding: utf-8

import torch


class BinaryF1Metric:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def __call__(self, predictions, targets):
        pred_labels = (predictions > 0).squeeze(dim=1)

        tp = pred_labels[targets == 1].sum().item()
        fp = pred_labels[targets == 0].sum().item()
        tn = torch.logical_not(pred_labels[targets == 0]).sum().item()
        fn = torch.logical_not(pred_labels[targets == 1]).sum().item()

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def get_precision(self):
        # TP / (TP + FP)
        num_pred_pos = self.tp + self.fp
        num_target_pos = self.tp + self.fn
        if num_pred_pos == 0:
            # There are no pixels that are predicted positives
            # and there no pixels to be predicted as positive
            return 1
        else:
            return self.tp / (self.tp + self.fp)

    def get_recall(self):
        # TP / (TP + FN)
        num_pred_pos = self.tp + self.fp
        num_target_pos = self.tp + self.fn
        if num_target_pos == 0:
            # There are no positive labels
            return 1
        else:
            return self.tp / (self.tp + self.fn)

    def get_value(self):
        return 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn)


class BinaryConfusionMatrixMetric:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def __call__(self, predictions, targets):
        pred_labels = (predictions > 0).squeeze(dim=1)

        tp = pred_labels[targets == 1].sum().item()
        fp = pred_labels[targets == 0].sum().item()
        tn = torch.logical_not(pred_labels[targets == 0]).sum().item()
        fn = torch.logical_not(pred_labels[targets == 1]).sum().item()

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def get_precision(self):
        # TP / (TP + FP)
        num_pred_pos = self.tp + self.fp
        num_target_pos = self.tp + self.fn
        if num_pred_pos == 0:
            # There are no pixels that are predicted positives
            # and there no pixels to be predicted as positive
            return 1
        else:
            return self.tp / (self.tp + self.fp)

    def get_recall(self):
        # TP / (TP + FN)
        num_pred_pos = self.tp + self.fp
        num_target_pos = self.tp + self.fn
        if num_target_pos == 0:
            # There are no positive labels
            return 1
        else:
            return self.tp / (self.tp + self.fn)

    def get_value(self):
        num_gt_N = self.tn + self.fp
        num_gt_P = self.fn + self.tp
        return [
            [
                self.tn / num_gt_N if num_gt_N > 0 else 1,
                self.fp / num_gt_N if num_gt_N > 0 else 1,
            ],
            [
                self.fn / num_gt_P if num_gt_P > 0 else 1,
                self.tp / num_gt_P if num_gt_P > 0 else 1,
            ],
        ]
