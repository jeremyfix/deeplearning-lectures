# coding: utf-8

# Standard imports
import inspect
import os
import pathlib

# External imports
import torch
import torch.nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

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
        device,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath_pt = savepath / "best_model.pt"
        self.savepath_onnx = savepath / "best_model.onnx"

        T, B = 2, 3
        self.dummy_inputs = torch.zeros((T, B, self.model.n_mels), device=device)

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

            # # Save also the onnx
            # # Create a wrapper that accepts unpacked tensors and unpacks the output
            # # This is necessary because torch.onnx.export uses symbolic tracing
            # # which doesn't support PackedSequence operations
            # class ExportableWrapper(torch.nn.Module):
            #     def __init__(self, model):
            #         super().__init__()
            #         self.model = model
            #
            #     def forward(self, x):
            #         # x is a regular tensor of shape (T, B, features)
            #         # we pack it to do a forward pass through the model
            #         T = x.shape[0]
            #         packed_x = pack_padded_sequence(x, lengths=[T]*x.shape[1])
            #
            #         packed_output = self.model(x)
            #         unpacked_output, _ = pad_packed_sequence(packed_output)
            #
            #         return unpacked_output
            #
            # wrapper = ExportableWrapper(self.model)
            #
            # # Use lower-level API to avoid torch.export.export issues
            # try:
            #     torch.onnx.export(
            #         wrapper,
            #         self.dummy_inputs,
            #         self.savepath_onnx,
            #         input_names=["input"],
            #         output_names=["output"],
            #         dynamic_shapes=(
            #             {0: "seq_len", 1: "batch"},
            #             {0: "seq_len", 1: "batch"},
            #         ),
            #     )
            # except Exception as e:
            #     # If ONNX export fails, we can still continue training
            #     # The model will be saved in PyTorch format
            #     import warnings
            #     warnings.warn(f"ONNX export failed: {e}. Continuing with PyTorch format only.")

            self.best_score = score

            # Switch the model back to its training state
            self.model.train(training)

            return True
        return False

# @SOL
def test_export():
    import tempfile

    # Dummy model
    class DummyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.n_mels = 20

        def forward(self, x):
            return x

    model = DummyModel()
       
    with tempfile.TemporaryDirectory() as tmpdirname:
        checkpoint = ModelCheckpoint(model, pathlib.Path(tmpdirname), torch.device('cpu'), min_is_best=True)
        checkpoint.update(1.0)
# SOL@

if __name__ == "__main__":
    # @TEMPL@ pass
    test_export() # @SOL@
