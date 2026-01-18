# coding: utf-8

# External imports
import torch
import torch.nn as nn

# Local imports
from . import encoding

class RealNGP(nn.Module):
    """
    Real valued Neural Graphic Primitive

    The first layer involves an Encoding then followed
    by dense layers to get the value of the function at the provided
    input coordinates.

    Arguments:
        dim_input (int): Number of input coordinates (e.g. 3 for x, y, t)
        dim_output (int): Number of output values (e.g. 1 for a scalar field)
        cfg (dict): Configuration for the encoding and MLP
    """

    def __init__(self, 
                 dim_input: int, 
                 dim_output: int, 
                 cfg: dict):
        super().__init__()

        # The input layer uses a hash encoding
        self.encoder = encoding.build_encoder(dim_input=dim_input, cfg=cfg["pos_encoding"])

        # Determine the dimensions of the embedding by doing a forward pass
        # through the encoder
        fake_input_size = (1, dim_input)
        fake_input = torch.zeros(*fake_input_size)
        output_encoding = self.encoder(fake_input) 
        output_encoding_size = output_encoding.shape[1] 

        # And then comes the FFNN with dense layers based
        # on the above coordinate encoding
        n_hidden_units = cfg["n_hidden_units"]
        n_hidden_layers = cfg["n_hidden_layers"]
        hidden_activation = nn.ReLU

        layers = []
        input_dim = output_encoding_size
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(input_dim, n_hidden_units))
            layers.append(hidden_activation())
            input_dim = n_hidden_units

        # The last dense layer projects onto the output space
        layers.append(nn.Linear(input_dim, dim_output))

        self.ffnn = nn.Sequential(*layers)

        # We could cascade the two steps
        # But we don't because we may want to skip the encoding
        # self.model = nn.Sequential(self.encoder, self.ffnn)

    def forward(self, x, skip_encoding=False):
        """
        Do a forward pass through the NIR. 

        Arguments:
            x (torch.Tensor): (K, d_in) tensor of coordinates
                where to evaluate the representation, where d_in is the dimensionality
                of the input and K is the number to points to be evaluated
                or of shape (K, output_encoding_size) if skip_encoding is True.
        """
    
        # We may skip the encoding
        # for example, when these have been pre-computed
        if not skip_encoding:
            x = self.encoder(x)

        x = self.ffnn(x)

        return x

