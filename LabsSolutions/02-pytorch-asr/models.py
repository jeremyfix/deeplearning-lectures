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
        #@TEMPL@self.cnn = None
        #@SOL
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=(41, 11),
                      stride=2,
                      padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(21, 11),
                      stride=(2, 1),
                      padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout2d(dropout)
        )
        #SOL@

        if cell_type not in ["GRU", "LSTM"]:
            raise NotImplementedError(f"Unrecognized cell type {cell_type}")

        cell_builder = getattr(nn, cell_type)

        # The temporal layers
        #@TEMPL@self.rnn = None
        #@SOL
        self.rnn = cell_builder(32*n_mels//2,
                                nhidden_rnn,
                                num_layers=nlayers_rnn,
                                bidirectional=True)
        #SOL@

        # The classification output layers
        #@TEMPL@self.charlin = None
        #@SOL
        self.charlin = nn.Sequential(
            nn.Linear(2*self.nhidden_rnn,
                      charmap.vocab_size)  # the vocabulary contrains the blank
        )
        #SOL@

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

        #@TEMPL@unpacked_inputs, lens_inputs = None
        unpacked_inputs, lens_inputs = pad_packed_sequence(inputs)  #@SOL@

        # Step 1b : Transform the unpackad input from (T, B, MELS)
        #           to (B, C=1, T, MELS) for treating this input
        #           as a 2D tensor with 1 channel (the power)
        #           Usefull functions : Tensor.transpose, Tensor.unsqueeze
        #           (1 line)

        #@TEMPL@unpacked_inputs = None
        unpacked_inputs = unpacked_inputs.transpose(0, 1).unsqueeze(dim=1)  #@SOL@

        ##
        # Step 2 : Make the forward pass through the convolutional part

        #@TEMPL@out_cnn = None
        out_cnn = self.cnn(unpacked_inputs)  #@SOL@

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
        #@TEMPL@out_cnn = None
        out_cnn = out_cnn.permute(2, 0, 1, 3).reshape(To, B, -1)  #@SOL@

        # Step 3b: You need to pack your padded tensors. Be carefull with
        #          the lengths attribute. It must be equal to the downscaled
        #          lenghts of the original signals. (1 line)

        #@TEMPL@rnn_inputs = None
        rnn_inputs = pack_padded_sequence(out_cnn, lengths=lens_inputs//4)  #@SOL@

        ##
        # Step 4 : Make the forward pass through the temporal layers
        #          The output tensor is (T, B, num_features).
        #          (1 line)

        #@TEMPL@packed_outrnn, _ = None
        packed_outrnn, _ = self.rnn(rnn_inputs)  #@SOL@

        ##
        # Step 5 : Classification output
        # Step 5a : Prepare your data by unpacking the output (1 line)

        #@TEMPL@unpacked_outrnn, lens_outrnn = None
        unpacked_outrnn, lens_outrnn = pad_packed_sequence(packed_outrnn)  #@SOL@

        # Step 5b : Make the forward pass through the classification output
        #           layers (1 line)

        #@TEMPL@out_lin = None
        out_lin = self.charlin(unpacked_outrnn)  #@SOL@

        # Step 5c : pack the output (1 line)

        #@TEMPL@outputs = None
        outputs = pack_padded_sequence(out_lin, lengths=lens_outrnn)  #@SOL@

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

    #@SOL
    def beam_decode(self,
                    inputs: PackedSequence,
                    beam_size: int,
                    blank_id: int):
        """
        Performs inference for the given output probabilities.
        Assuming a single sample is given.
        Adapted from :
            https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0

        Arguments:
            inputs (PackedSequence) : the input spectrogram
            beam_size (int): Size of the beam to use during inference.
            blank (int): Index of the CTC blank label.
        Returns the output label sequence and the corresponding negative
        log-likelihood estimated by the decoder.
        """
        NEG_INF = -float("inf")
        def make_new_beam():
            fn = lambda : (NEG_INF, NEG_INF)
            return collections.defaultdict(fn)

        def logsumexp(*args):
            """
            Stable log sum exp.
            """
            if all(a == NEG_INF for a in args):
                return NEG_INF
            a_max = max(args)
            lsp = math.log(sum(math.exp(a - a_max) for a in args))
            return a_max + lsp

        with torch.no_grad():
            outputs = self.forward(inputs)  # packed T, B, num_char
            unpacked_outputs, lens_outputs = pad_packed_sequence(outputs)
            T, batch_size, S = unpacked_outputs.shape
            if batch_size != 1:
                raise NotImplementedError("Can decode only one batch at a time")

            unpacked_outputs = unpacked_outputs.squeeze(dim=1)  # seq, vocab_size
            probs = unpacked_outputs.log_softmax(dim=1)

            # Elements in the beam are (prefix, (p_blank, p_no_blank))
            # Initialize the beam with the empty sequence, a probability of
            # 1 for ending in blank and zero for ending in non-blank
            # (in log space).
            beam = [(tuple(), (0.0, NEG_INF))]

            for t in tqdm.tqdm(range(T)):  # Loop over time

                # A default dictionary to store the next step candidates.
                next_beam = make_new_beam()

                for s in range(S): # Loop over vocab
                    p = probs[t, s]

                    # The variables p_b and p_nb are respectively the
                    # probabilities for the prefix given that it ends in a
                    # blank and does not end in a blank at this time step.
                    for prefix, (p_b, p_nb) in beam:  # Loop over beam

                        # If we propose a blank the prefix doesn't change.
                        # Only the probability of ending in blank gets updated.
                        if s == blank_id:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)
                            continue

                        # Extend the prefix by the new character s and add it to
                        # the beam. Only the probability of not ending in blank
                        # gets updated.
                        end_t = prefix[-1] if prefix else None
                        n_prefix = prefix + (s,)
                        n_p_b, n_p_nb = next_beam[n_prefix]
                        if s != end_t:
                            n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                        else:
                            # We don't include the previous probability of not ending
                            # in blank (p_nb) if s is repeated at the end. The CTC
                            # algorithm merges characters not separated by a blank.
                            n_p_nb = logsumexp(n_p_nb, p_b + p)

                        # *NB* this would be a good place to include an LM score.
                        next_beam[n_prefix] = (n_p_b, n_p_nb)

                        # If s is repeated at the end we also update the unchanged
                        # prefix. This is the merging case.
                        if s == end_t:
                            n_p_b, n_p_nb = next_beam[prefix]
                            n_p_nb = logsumexp(n_p_nb, p_nb + p)
                            next_beam[prefix] = (n_p_b, n_p_nb)

                # Sort and trim the beam before moving on to the
                # next time-step.
                beam = sorted(next_beam.items(),
                              key=lambda x: logsumexp(*x[1]),
                              reverse=True)
                beam = beam[:beam_size]

            best = beam[0]

            # Decode the list of integers
            seq = self.charmap.decode(best[0])

            return [(-logsumexp(*best[1]), seq)]
    #SOL@

#@SOL
def ex_ctc():
    # The size of our minibatches
    batch_size = 32
    # The size of our vocabulary (including the blank character)
    vocab_size = 44
    # The class id for the blank token
    blank_id = 43

    max_spectro_length = 50
    min_transcript_length = 10
    max_transcript_length = 30

    # Compute a dummy vector of probabilities over the vocabulary (including the blank)
    # log_probs is here batch_first, i.e. (Batch, Tx, vocab_size)
    log_probs = torch.randn(batch_size, max_spectro_length, vocab_size).log_softmax(dim=1)
    spectro_lengths = torch.randint(low=max_transcript_length,
                                    high=max_spectro_length,
                                    size=(batch_size, ))

    # Compute some dummy transcripts
    # targets is here (batch_size, Ty)
    targets = torch.randint(low=0, high=vocab_size+1,  # include the blank character
                            size=(batch_size, max_transcript_length))
    target_lengths = torch.randint(low=min_transcript_length,
                                   high=max_transcript_length,
                                   size=(batch_size, ))
    loss = torch.nn.CTCLoss(blank=blank_id)

    # The log_probs must be given as (Tx, Batch, vocab_size)
    vloss = loss(log_probs.transpose(0, 1),
                 targets,
                 spectro_lengths,
                 target_lengths)

    print(f"Our dummy loss equals {vloss}")

def ex_pack():
    import random
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

    batch_size = 10
    n_mels = 80
    max_length = 512
    # Given a list of batch_size tensors of variable sizes of shape (Ti, n_mels)
    tensors = [torch.randn(random.randint(1, max_length), n_mels)for i in range(batch_size)]

    # To be packed, the tensors need to be sorted by
    # decreasing length
    tensors = sorted(tensors,
                     key=lambda tensor: tensor.shape[0],
                     reverse=True)
    lengths = [t.shape[0] for t in tensors]

    # We start by padding the sequences to the max_length
    tensors = pad_sequence(tensors, batch_first=True)
    # tensors is (batch_size, T, n_mels)
    # note T is equal to the maximal length of the sequences

    # We can pack the sequence
    packed_data = pack_padded_sequence(tensors, lengths=lengths,
                                      batch_first=True)

    # Latter, we can unpack the sequence
    unpacked_data, lens_data = pad_packed_sequence(packed_data,
                                                  batch_first=True)



if __name__ == '__main__':
    # ex_ctc()
    # ex_pack()
    test_model()
#SOL@
