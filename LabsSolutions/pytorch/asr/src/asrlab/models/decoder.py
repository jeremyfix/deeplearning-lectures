# coding: utf-8

# Standard imports
import collections
import math
import tqdm

# External imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence

def greedy_decode(packed_outputs: PackedSequence, charmap):
    """
    Greedy decoder.

    Args:
        inputs (PackedSequence) : the input spectrogram

    Returns:
        list of pairs of the negative log likelihood of the sequence
                      with the corresponding sequence
    """
    with torch.no_grad():
        unpacked_outputs, lens_outputs = pad_packed_sequence(packed_outputs)
        seq_len, batch_size, num_char = unpacked_outputs.shape

        if batch_size != 1:
            raise NotImplementedError("Can decode only one batch at a time")

        unpacked_outputs = unpacked_outputs.squeeze(dim=1)  # seq, vocab_size
        outputs = unpacked_outputs.log_softmax(dim=1)
        top_values, top_indices = outputs.topk(k=1, dim=1)

        # We look for a eos token
        eos_token = charmap.eos
        eos_pos = None
        for ic, token in enumerate(top_indices):
            if token == eos_token:
                eos_pos = ic
        if eos_pos is None:
            neg_log_prob = -top_values.sum()
            seq = [c for c in top_indices]
        else:
            neg_log_prob = -top_values[: (eos_pos + 1)].sum()
            seq = [c for c in top_indices[: (eos_pos + 1)]]

        # Remove the repetitions
        if len(seq) != 0:
            last_char = seq[-1]
            seq = [c1 for c1, c2 in zip(seq[:-1], seq[1:]) if c1 != c2]
            seq.append(last_char)

        # Remove the blank
        seq = [c for c in seq if c != charmap.blankid]

        # Decode the list of integers
        seq = charmap.decode(seq)

        return [(neg_log_prob, seq)]

def beam_decode(packed_outputs: PackedSequence, beam_size: int, blank_id: int, charmap):
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
        fn = lambda: (NEG_INF, NEG_INF)
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
        unpacked_outputs, lens_outputs = pad_packed_sequence(packed_outputs)
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

            for s in range(S):  # Loop over vocab
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
            beam = sorted(
                next_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True
            )
            beam = beam[:beam_size]

        best = beam[0]

        # Decode the list of integers
        seq = charmap.decode(best[0])

        return [(-logsumexp(*best[1]), seq)]
