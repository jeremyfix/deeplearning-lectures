#!/bin/bash

NB_RUNS=10

for iter in $(seq 1 $NB_RUNS)
do
    echo "Run $iter"
    python3 train.py --model linear
    python3 train.py --model fc --minimize --dropout --L2
    python3 train.py --model vanilla --normalize --dropout --L2
    python3 train.py --model fancy --normalize --dropout --L2
    python3 train.py --model fancy --normalize --dropout --L2 --data_augment
done
