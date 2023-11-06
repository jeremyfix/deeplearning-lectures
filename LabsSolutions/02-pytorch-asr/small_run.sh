#!/usr/bin/env bash

python main_ctc.py --debug train --nhidden_rnn 64 --nlayers_rnn 1  --batch_size 2 --cell_type LSTM
