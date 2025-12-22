
import torch


def ex_ctc():
    # The size of our minibatches
    batch_size = 32
    # The size of our vocabulary (including the blank character)
    vocab_size = 44
    # The class id for the blank token (we take the default of the CTC loss)
    blank_id = 0

    max_spectro_length = 50
    min_transcript_length = 10
    max_transcript_length = 30

    # Compute a dummy vector of probabilities over the vocabulary (including the blank)
    # log_probs is (Tx, Batch, vocab_size)
    log_probs = torch.randn(max_spectro_length, batch_size, vocab_size).log_softmax(dim=2)
    spectro_lengths = torch.randint(low=max_transcript_length,
                                    high=max_spectro_length,
                                    size=(batch_size, ))

    # Compute some dummy transcripts
    # longer than necessary. The true length of the transcripts is given
    # in the target_lengths tensor
    # targets is here (batch_size, Ty)
    targets = torch.randint(low=0, high=vocab_size,  # include the blank character
                            size=(batch_size, max_transcript_length))
    target_lengths = torch.randint(low=min_transcript_length,
                                   high=max_transcript_length,
                                   size=(batch_size, ))
    loss = torch.nn.CTCLoss(blank=blank_id)

    # The log_probs must be given as (Tx, Batch, vocab_size)
    vloss = loss(log_probs,
                 targets,
                 spectro_lengths,
                 target_lengths)

    print(f"Our dummy loss equals {vloss}")

if __name__ == '__main__':
    ex_ctc()
