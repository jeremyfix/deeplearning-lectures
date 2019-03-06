import os
import torch
from torch.nn.modules.module import _addindent

def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

class ModelCheckpoint:

    def __init__(self, filepath, dict_to_save):
        self.min_loss = None
        self.filepath = filepath
        self.dict_to_save = dict_to_save

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            #torch.save(self.model.state_dict(), self.filepath)
            torch.save(self.dict_to_save, self.filepath)
            self.min_loss = loss

# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
def prod_dim(dims: torch.Size):
    p = 1
    for s in dims:
        p = p*s
    return p

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([prod_dim(p.size()) for p in module.parameters()])
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    tmpstr += '\n {} learnable parameters'.format(total_params)
    return tmpstr


