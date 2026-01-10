# coding: utf-8
from __future__ import annotations

# Standard imports
import collections
import math
from typing import List, Tuple
import tqdm

# External imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

# Local imports
from asrlab import data
from . import decoder

class LinearModel(nn.Module):
    """
    Dummy Model, just a linear layer
    """

    def __init__(
        self,
        charmap: data.CharMap,
        n_mels: int,
    ) -> None:
        """
        Args:
            charmap (data.Charmap) : the character/int map
            n_mels (int) : number of input mel scales
        """
        super(LinearModel, self).__init__()
        self.charmap = charmap
        self.n_mels = n_mels

        self.charlin = nn.Sequential(
            nn.Linear(
                self.n_mels, charmap.vocab_size
            )  # the vocabulary contrains the blank
        )

    def forward(self, inputs: PackedSequence) -> PackedSequence:

        # (T, B, MELS)
        unpacked_inputs, lens_inputs = pad_packed_sequence(inputs)
        out_lin = self.charlin(unpacked_inputs)  # (T, B, num_out)
        outputs = pack_padded_sequence(out_lin, lengths=lens_inputs)

        return outputs

    def decode(self, inputs: PackedSequence) -> List[Tuple[float, str]]:
        with torch.no_grad():
            outputs = self.forward(inputs)
        return decoder.greedy_decode(outputs, self.charmap)

    def beam_decode(self, inputs: PackedSequence, beam_size: int, blank_id: int):
        with torch.no_grad():
            outputs = self.forward(inputs)
        return decoder.beam_decode(outputs, beam_size, blank_id, self.charmap)


# @SOL
class CTCModelExportable(nn.Module):
    """
    Wrapper around CTCModel that accepts dense (unpacked) tensors instead of
    PackedSequences. This allows the model to be exported to ONNX format.
    
    This class is designed for export purposes only and should not be used
    for training. It accepts inputs of shape (T, B, n_mels) and returns
    outputs of shape (T, B, vocab_size).
    """

    def __init__(self, model: CTCModel) -> None:
        """
        Args:
            model: A trained CTCModel instance
        """
        super().__init__()
        self.model = model
        self.n_mels = model.n_mels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dense tensors.
        
        Args:
            inputs: Tensor of shape (T, B, n_mels)
        
        Returns:
            Tensor of shape (T, B, vocab_size)
        """
        # Get the batch size and sequence length
        T, B, _ = inputs.shape
        
        # Create a PackedSequence from the dense tensor
        # Assume all sequences have full length T
        packed_inputs = pack_padded_sequence(inputs, lengths=[T] * B)
        
        # Forward pass through the model
        packed_output = self.model(packed_inputs)
        
        # Unpack the output to get dense tensor
        unpacked_output, _ = pad_packed_sequence(packed_output)
        
        return unpacked_output
# SOL@

class CTCModel(nn.Module):
    """
    Connectionist Temporal Classification architecture inspired by
    the DeepSpeech2. To be used with the CTC Loss
    """

    def __init__(
        self,
        charmap: data.CharMap,
        n_mels: int,
        nhidden_rnn: int,
        nlayers_rnn: int,
        cell_type: str,
        dropout: float,
    ) -> None:
        """
        Args:
            charmap (data.Charmap) : the character/int map
            n_mels (int) : number of input mel scales
            nhidden_rnn (int): number of LSTM cells per layer and per direction
            nlayers_rnn (int) : number of stacked RNN layers
            cell_type(str) either "GRU" or "LSTM"
            dropout(float): the amount of dropout in the feedforward layers
        """
        super().__init__()
        self.charmap = charmap
        self.n_mels = n_mels
        self.nhidden_rnn = nhidden_rnn
        self.nlayers_rnn = nlayers_rnn
        self.cell_type = cell_type

        ###########################
        #### START CODING HERE ####
        ###########################

        # The convolutional layers
        # @TEMPL@self.cnn = None
        # @SOL
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(41, 11),
                stride=2,
                padding=(20, 5),
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout2d(dropout),
        )
        # SOL@

        if cell_type not in ["GRU", "LSTM"]:
            raise NotImplementedError(f"Unrecognized cell type {cell_type}")

        cell_builder = getattr(nn, cell_type)

        # The temporal layers
        # @TEMPL@self.rnn = None
        # @SOL
        self.rnn = cell_builder(
            32 * n_mels // 2, nhidden_rnn, num_layers=nlayers_rnn, bidirectional=True
        )
        # SOL@

        # The classification output layers
        # @TEMPL@self.charlin = None
        # @SOL
        self.charlin = nn.Sequential(
            nn.Linear(
                2 * self.nhidden_rnn, charmap.vocab_size
            )  # the vocabulary contrains the blank
        )
        # SOL@

        ##########################
        #### STOP CODING HERE ####
        ##########################

        # self.reset_parameters()

    def reset_parameters(self):
        # Let us initialize the biases with :
        # - a high forget bias
        # - a zero input bias
        # - a low output bias
        with torch.no_grad():
            if self.cell_type == "LSTM":
                for i in range(self.nlayers_rnn):
                    forw_gates = getattr(self.rnn, f"bias_ih_l{i}").chunk(4, dim=0)
                    iig, ifg, igg, iog = forw_gates
                    iig.fill_(0.0)
                    ifg.fill_(1.0)
                    igg.fill_(0.0)
                    iog.fill_(0.0)

                    forw_gates = getattr(self.rnn, f"bias_hh_l{i}").chunk(4, dim=0)
                    hig, hfg, hgg, hog = forw_gates
                    hig.fill_(0.0)
                    hfg.fill_(0.0)
                    hgg.fill_(0.0)
                    hog.fill_(0.0)

                    rev_gates = getattr(self.rnn, f"bias_ih_l{i}_reverse").chunk(
                        4, dim=0
                    )
                    iig, ifg, igg, iog = rev_gates
                    iig.fill_(0.0)
                    ifg.fill_(1.0)
                    igg.fill_(0.0)
                    iog.fill_(0.0)

                    rev_gates = getattr(self.rnn, f"bias_hh_l{i}_reverse").chunk(
                        4, dim=0
                    )
                    hig, hfg, hgg, hog = rev_gates
                    hig.fill_(0.0)
                    hfg.fill_(0.0)
                    hgg.fill_(0.0)
                    hog.fill_(0.0)
            else:
                # GRU
                for i in range(self.nlayers_rnn):
                    for direction in ["", "_reverse"]:
                        gates = getattr(self.rnn, f"bias_ih_l{i}{direction}").chunk(
                            3, dim=0
                        )
                        irg, izg, ing = gates
                        irg.fill_(1.0)
                        izg.fill_(-1.0)
                        ing.fill_(0.0)

                        gates = getattr(self.rnn, f"bias_hh_l{i}{direction}").chunk(
                            3, dim=0
                        )
                        hrg, hzg, hng = gates
                        hrg.fill_(0.0)
                        hzg.fill_(0.0)
                        hng.fill_(0.0)

    def forward(self, inputs: PackedSequence) -> PackedSequence:

        ###########################
        #### START CODING HERE ####
        ###########################

        ##
        # Step 1 : Prepare your data for going through the convolutional
        #          layers. You need to unpack your data and transform the
        #          tensor from (T, B, MELS) to (B, C=1, T, MELS)
        # Step 1a : Unpack your data. Once unpacked, unpacked_inputs
        #           is of shape (T, B, MELS)
        #           (1 line)
        # @TEMPL@unpacked_inputs, lens_inputs = None
        unpacked_inputs, lens_inputs = pad_packed_sequence(inputs)  # @SOL@

        # Step 1b : Transform the unpackad input from (T, B, MELS)
        #           to (B, C=1, T, MELS) for treating this input
        #           as a 2D tensor with 1 channel (the power)
        #           Usefull functions : Tensor.transpose, Tensor.unsqueeze
        #           (1 line)
        # @TEMPL@unpacked_inputs = None
        unpacked_inputs = unpacked_inputs.transpose(0, 1).unsqueeze(dim=1)  # @SOL@

        ##
        # Step 2 : Make the forward pass through the convolutional part
        # @TEMPL@out_cnn = None
        out_cnn = self.cnn(unpacked_inputs)  # @SOL@

        ##
        # Step 3 : Prepare your data for going through the temporal
        #          layers.
        # Step 3a: You need to transform your tensors from
        #          (B, C, To, W) to (To,B,C*W) with To=T//s1, W=MELS//s2
        #          and s1, s2 the downsampling factors along the temporal
        #          and frequency dimensions
        #          Useful functions : Tensor.permute , Tensor.reshape
        #          Hint : reshape accept the special value -1 for "as needed"
        #          (1 line)
        B = out_cnn.shape[0]
        To = out_cnn.shape[2]
        # @TEMPL@out_cnn = None
        out_cnn = out_cnn.permute(2, 0, 1, 3).reshape(To, B, -1)  # @SOL@

        # Step 3b: You need to pack your padded tensors. Be carefull with
        #          the lengths attribute. It must be equal to the downscaled
        #          lenghts of the original signals. (1 line)
        # @TEMPL@rnn_inputs = None
        rnn_inputs = pack_padded_sequence(out_cnn, lengths=lens_inputs // 4)  # @SOL@

        ##
        # Step 4 : Make the forward pass through the temporal layers
        #          The output tensor is (T, B, num_features).
        #          (1 line)
        # @TEMPL@packed_outrnn, _ = None
        packed_outrnn, _ = self.rnn(rnn_inputs)  # @SOL@

        ##
        # Step 5 : Classification output
        # Step 5a : Prepare your data by unpacking the output (1 line)
        # @TEMPL@unpacked_outrnn, lens_outrnn = None
        unpacked_outrnn, lens_outrnn = pad_packed_sequence(packed_outrnn)  # @SOL@

        # Step 5b : Make the forward pass through the classification output
        #           layers (1 line)
        # @TEMPL@out_lin = None
        # @SOL
        out_lin = self.charlin(unpacked_outrnn)  # (T, B, num_out)
        # SOL@

        # Step 5c : pack the output (1 line)
        # @TEMPL@outputs = None
        outputs = pack_padded_sequence(out_lin, lengths=lens_outrnn)  # @SOL@

        ##########################
        #### STOP CODING HERE ####
        ##########################

        return outputs

    def decode(self, inputs: PackedSequence) -> List[Tuple[float, str]]:
        with torch.no_grad():
            outputs = self.forward(inputs)
        return decoder.greedy_decode(outputs, self.charmap)

    def beam_decode(self, inputs: PackedSequence, beam_size: int, blank_id: int):
        with torch.no_grad():
            outputs = self.forward(inputs)
        return decoder.beam_decode(outputs, beam_size, blank_id, self.charmap)

# @SOL
class CTCModelExportable(nn.Module):
    """
    Wrapper around CTCModel components for ONNX export that uses dense tensors
    instead of PackedSequence. This class reconstructs the model architecture
    without pack_padded_sequence/pad_packed_sequence operations which are not
    supported by ONNX export.
    
    This class is designed for export purposes only and should not be used
    for training. It accepts inputs of shape (T, B, n_mels) and returns
    outputs of shape (T, B, vocab_size).
    """

    def __init__(self, model: "CTCModel") -> None:
        """
        Args:
            model: A trained CTCModel instance
        """
        super().__init__()
        # Copy the CNN layers
        self.cnn = model.cnn
        # Copy the RNN layers
        self.rnn = model.rnn
        # Copy the classification layers
        self.charlin = model.charlin
        self.n_mels = model.n_mels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dense tensors (without PackedSequence).
        
        Args:
            inputs: Tensor of shape (T, B, n_mels)
        
        Returns:
            Tensor of shape (T_out, B, vocab_size) where T_out = T // 4
        """
        # Step 1: Unpack inputs and transform from (T, B, MELS) to (B, C=1, T, MELS)
        unpacked_inputs = inputs.transpose(0, 1).unsqueeze(dim=1)

        # Step 2: Pass through convolutional layers
        out_cnn = self.cnn(unpacked_inputs)  # (B, C, To, W)

        # Step 3: Transform from (B, C, To, W) to (To, B, C*W)
        B = out_cnn.shape[0]
        To = out_cnn.shape[2]  # This is the time dimension after downsampling
        out_cnn = out_cnn.permute(2, 0, 1, 3).reshape(To, B, -1)

        # Step 4: Pass through RNN layers
        # Note: The RNN now operates on the downsampled sequence
        # In the original model this happens implicitly when creating PackedSequence
        rnn_output, _ = self.rnn(out_cnn)  # (To, B, num_hidden*2)

        # Step 5: Pass through classification layers
        print("Step 5")
        out_lin = self.charlin(rnn_output)  # (To, B, vocab_size)
        print("All done")
        return out_lin
# SOL@

