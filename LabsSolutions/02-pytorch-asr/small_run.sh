#!/usr/bin/env bash

# BS 2
# python main_ctc.py --debug train --nhidden_rnn 1024 --nlayers_rnn 4  --batch_size 2 --cell_type GRU --base_lr 0.001 --num_epochs 500 --grad_clip 10

# BS 2 : plus gros réseau, overfit plus vite
# python main_ctc.py --debug train --nhidden_rnn 1024 --nlayers_rnn 4  --batch_size 2 --cell_type GRU --base_lr 0.0003 --num_epochs 500 --grad_clip 10

# BS 4
# python main_ctc.py --debug train --nhidden_rnn 1024 --nlayers_rnn 4  --batch_size 4 --cell_type GRU --base_lr 0.0005 --num_epochs 500 --grad_clip 10


# BS 32
# python main_ctc.py --debug --nhidden_rnn 1024 --nlayers_rnn 4  --batch_size 32 --cell_type GRU --base_lr 0.0001 --num_epochs 500 --grad_clip 5 train

# not necessarily better than above but ends overfitting, say, in 400 epochs
# python main_ctc.py --debug --nhidden_rnn 1024 --nlayers_rnn 4  --batch_size 32 --cell_type GRU --base_lr 0.0001 --num_epochs 500 --dropout 0.0 --weight_decay 0.0 --grad_clip 10 train


# Overfit du training set !! ctc_40469_0
python main_ctc.py --nhidden_rnn 1024 --nlayers_rnn 4  --batch_size 128 --cell_type GRU --base_lr 0.001 --num_epochs 500 --dropout 0.0 --weight_decay 0.0 --grad_clip 20 train

# On relance
# python main_ctc.py --nhidden_rnn 1024 --nlayers_rnn 4  --batch_size 128 --cell_type GRU --base_lr 0.001 --num_epochs 500 --dropout 0.0 --weight_decay 0.0 --grad_clip 20 train

