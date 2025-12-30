# coding: utf-8

# Standard imports
import inspect
import os

# External imports
import torch
import torch.nn

class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def tab(n):
    return " " * 4 * n


def fail(msg):
    print(
        colors.FAIL
        + tab(1)
        + f"[FAILED] From {inspect.stack()[1][3]}"
        + msg
        + colors.ENDC
    )


def succeed(msg=""):
    print(colors.OKGREEN + tab(1) + "[PASSED]" + msg + colors.ENDC)


def head(msg):
    print(colors.HEADER + msg + colors.ENDC)


def info(msg):
    print(colors.OKBLUE + tab(1) + msg + colors.ENDC)


def test_equal(l1, l2, eps):
    return all([abs(l1i - l2i) <= eps for l1i, l2i in zip(l1, l2)])

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
        savepath,
        input_size,
        device,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath_pt = savepath / "best_model.pt"
        self.savepath_onnx = savepath / "best_model.onnx"
        self.dummy_inputs = torch.zeros(input_size, device=device)
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

            # Save also the onnx
            torch.onnx.export(
                self.model,
                self.dummy_inputs,
                self.savepath_onnx,
                input_names=["scan"],
                output_names=["output"],
                dynamic_axes={
                    "scan": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                },
            )

            self.best_score = score

            # Switch the model back to its training state
            self.model.train(training)

            return True
        return False


