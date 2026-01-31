# coding: utf-8

# Standard imports
import os
import logging
import sys
import pathlib

# External imports
import torch
import torch.nn
import tqdm


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath: pathlib.Path,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath_pt = savepath / "best_model.pt"
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            # Keep record of whether the model was in training or eval
            training = self.model.training

            # Switch the model to eval mode before the export
            self.model.eval()

            # Export the pytorch parameters tensor
            torch.save(self.model.state_dict(), self.savepath_pt)

            self.best_score = score

            # Switch the model back to its training state
            self.model.train(training)

            return True
        return False


def train(model, loader, f_loss, optimizer, device, batch_metrics):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    for bname, bm in batch_metrics.items():
        bm.reset()

    for inputs, targets in tqdm.tqdm(loader):
        inputs = inputs.to(device)

        if isinstance(targets, tuple) or isinstance(targets, list):
            # For the MRI dataset, we have multiple targets
            targets = tuple(t.to(device) for t in targets)
        else:
            targets = targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        total_loss = f_loss(outputs, targets)
  
        if hasattr(model, "penalty"):
            # Combine loss and penalty in a single backward pass
            total_loss += model.penalty()

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update the metrics
        for bname, bm in batch_metrics.items():
            bm(outputs, targets)

    # Compute the value of the batch metrics
    tot_metrics = {}
    for bname, bm in batch_metrics.items():
        tot_metrics[bname] = bm.get_value()

    return tot_metrics


def test(model, loader, device, batch_metrics):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    for bname, bm in batch_metrics.items():
        bm.reset()

    for inputs, targets in tqdm.tqdm(loader):
        inputs = inputs.to(device)

        if isinstance(targets, tuple) or isinstance(targets, list):
            # For the MRI dataset, we have multiple targets
            targets = tuple(t.to(device) for t in targets)
        else:
            targets = targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        # Update the metrics
        for bname, bm in batch_metrics.items():
            bm(outputs, targets)

    # Compute the value of the batch metrics
    tot_metrics = {}
    for bname, bm in batch_metrics.items():
        tot_metrics[bname] = bm.get_value()

    return tot_metrics

def build_coordinate_Nd(*Ns, device=torch.device("cpu")):
    coords_lin = [torch.linspace(0, 1, Ni, device=device) for Ni in Ns]
    coords_mesh = torch.meshgrid(*coords_lin, indexing="ij")
    stacked = torch.stack(coords_mesh, -1).view(-1, len(Ns))
    return stacked

def test_coords():
    coords = build_coordinate_Nd(3, 4, 5, 6)
    print(coords.shape)
    print(coords)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    test_coords()
