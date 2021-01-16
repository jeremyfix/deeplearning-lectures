#!/usr/bin/env python3

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
import data


class CTCModel(nn.Module):
    """
    Connectionist Temporal Classification architecture inspired by
    the DeepSpeech2. To be used with the CTC Loss
    """

    def __init__(self,
                 charmap: data.CharMap,
                 n_mels: int,
                 nhidden_rnn: int,
                 nlayers_rnn: int,
                 cell_type: str,
                 dropout: float) -> None:
        """
        Args:
            charmap (data.Charmap) : the character/int map
            n_mels (int) : number of input mel scales
            nhidden_rnn (int): number of LSTM cells per layer and per direction
            nlayers_rnn (int) : number of stacked RNN layers
            cell_type(str) either "GRU" or "LSTM"
            dropout(float): the amount of dropout in the feedforward layers
        """
        super(CTCModel, self).__init__()
        self.charmap = charmap
        self.n_mels = n_mels
        self.nhidden_rnn = nhidden_rnn
        self.nlayers_rnn = nlayers_rnn
        self.cell_type = cell_type

        ###########################
        #### START CODING HERE ####
        ###########################

        # The convolutional layers
        self.cnn = None

        if cell_type not in ["GRU", "LSTM"]:
            raise NotImplementedError(f"Unrecognized cell type {cell_type}")

        cell_builder = getattr(nn, cell_type)

        # The temporal layers
        self.rnn = None

        # The classification output layers
        self.charlin = None

        ##########################
        #### STOP CODING HERE ####
        ##########################

        self.reset_parameters()

    def reset_parameters(self):
        # Let us initialize the biases with :
        # - a high forget bias
        # - a zero input bias
        # - a low output bias
        with torch.no_grad():
            if self.cell_type == "LSTM":
                for i in range(self.nlayers_rnn):
                    forw_gates = getattr(self.rnn,
                                         f'bias_ih_l{i}').chunk(4, dim=0)
                    iig, ifg, igg, iog = forw_gates
                    iig.fill_(0.)
                    ifg.fill_(1.)
                    igg.fill_(0.)
                    iog.fill_(0.)

                    forw_gates = getattr(self.rnn,
                                         f'bias_hh_l{i}').chunk(4, dim=0)
                    hig, hfg, hgg, hog = forw_gates
                    hig.fill_(0.)
                    hfg.fill_(0.)
                    hgg.fill_(0.)
                    hog.fill_(0.)

                    rev_gates = getattr(self.rnn,
                                        f'bias_ih_l{i}_reverse').chunk(4, dim=0)
                    iig, ifg, igg, iog = rev_gates
                    iig.fill_(0.)
                    ifg.fill_(1.)
                    igg.fill_(0.)
                    iog.fill_(0.)

                    rev_gates = getattr(self.rnn,
                                        f'bias_hh_l{i}_reverse').chunk(4, dim=0)
                    hig, hfg, hgg, hog = rev_gates
                    hig.fill_(0.)
                    hfg.fill_(0.)
                    hgg.fill_(0.)
                    hog.fill_(0.)
            else:
                # GRU
                for i in range(self.nlayers_rnn):
                    for direction in ['', '_reverse']:
                        gates = getattr(self.rnn,
                                        f'bias_ih_l{i}{direction}').chunk(3, dim=0)
                        irg, izg, ing = gates
                        irg.fill_(1.)
                        izg.fill_(-1.)
                        ing.fill_(0.)

                        gates = getattr(self.rnn,
                                        f'bias_hh_l{i}{direction}').chunk(3, dim=0) 
                        hrg, hzg, hng = gates
                        hrg.fill_(0.)
                        hzg.fill_(0.)
                        hng.fill_(0.)


    def forward(self,
                inputs: PackedSequence) -> PackedSequence:

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

        unpacked_inputs, lens_inputs = None

        # Step 1b : Transform the unpackad input from (T, B, MELS)
        #           to (B, C=1, T, MELS) for treating this input
        #           as a 2D tensor with 1 channel (the power)
        #           Usefull functions : Tensor.transpose, Tensor.unsqueeze
        #           (1 line)

        unpacked_inputs = None

        ##
        # Step 2 : Make the forward pass through the convolutional part

        out_cnn = None

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
        out_cnn = None

        # Step 3b: You need to pack your padded tensors. Be carefull with
        #          the lengths attribute. It must be equal to the downscaled
        #          lenghts of the original signals. (1 line)

        rnn_inputs = None

        ##
        # Step 4 : Make the forward pass through the temporal layers
        #          The output tensor is (T, B, num_features).
        #          (1 line)

        packed_outrnn, _ = None

        ##
        # Step 5 : Classification output
        # Step 5a : Prepare your data by unpacking the output (1 line)

        unpacked_outrnn, lens_outrnn = None

        # Step 5b : Make the forward pass through the classification output
        #           layers (1 line)

        out_lin = None

        # Step 5c : pack the output (1 line)

        outputs = None

        ##########################
        #### STOP CODING HERE ####
        ##########################

        return outputs

    def decode(self,
               inputs: PackedSequence) -> List[Tuple[float, str]]:
        """
        Greedy decoder.

        Args:
            inputs (PackedSequence) : the input spectrogram

        Returns:
            list of pairs of the negative log likelihood of the sequence
                          with the corresponding sequence
        """
        with torch.no_grad():
            outputs = self.forward(inputs)  # packed  T, B, num_char
            unpacked_outputs, lens_outputs = pad_packed_sequence(outputs)
            seq_len, batch_size, num_char = unpacked_outputs.shape

            if batch_size != 1:
                raise NotImplementedError("Can decode only one batch at a time")

            unpacked_outputs = unpacked_outputs.squeeze(dim=1)  # seq, vocab_size
            outputs = unpacked_outputs.log_softmax(dim=1)
            top_values, top_indices = outputs.topk(k=1, dim=1)

            # We look for a eos token
            eos_token = self.charmap.eos
            eos_pos = None
            for ic, token in enumerate(top_indices):
                if token == eos_token:
                    eos_pos = ic
            if eos_pos is None:
                neg_log_prob = -top_values.sum()
                seq = [c for c in top_indices]
            else:
                neg_log_prob = -top_values[:(eos_pos+1)].sum()
                seq = [c for c in top_indices[:(eos_pos+1)]]

            # Remove the repetitions
            if len(seq) != 0:
                last_char = seq[-1]
                seq = [c1 for c1, c2 in zip(seq[:-1], seq[1:]) if c1 != c2]
                seq.append(last_char)

            # Remove the blank 
            seq = [c for c in seq if c != self.charmap.blankid]

            # Decode the list of integers
            seq = self.charmap.decode(seq)

            return [(neg_log_prob, seq)]


