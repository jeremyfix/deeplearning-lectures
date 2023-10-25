#!/bin/bash

NB_RUNS=10

for iter in $(seq 1 $NB_RUNS)
do
    echo "Run $iter"
    python3 main.py --model linear train
    python3 main.py --model fc --minimize --dropout --L2 train
    python3 main.py --model vanilla --normalize --dropout --L2 train
    python3 main.py --model fancy --normalize --dropout --L2 train
    python3 main.py --model fancy --normalize --dropout --L2 --data_augment train
done
