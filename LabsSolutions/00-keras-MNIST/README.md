
To train the several types of models, here is an example usage 

    python3 train.py --model linear
    python3 train.py --model fc --minimize --dropout --L2
    python3 train.py --model vanilla --normalize --dropout --L2
    python3 train.py --model fancy --normalize --dropout --L2
    python3 train.py --model fancy --normalize --dropout --L2 --data_augment

You can monitor the progress with tensorboard, the logs being saved in ./logs

    tensorboard --logdir ./logs


To test all the models of the logs directory :

    python3 test.py --modelsdir ./logs

The average model is not implemented.
