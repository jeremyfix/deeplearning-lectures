I suggest you to train several models, you may need to adapt the parameters below

    # To see the options :
    python3 train.py 

    # e.g. run several times :
    python3 train.py --use_gpu --normalize --model fancyCNN

You can monitor the progress with tensorboard, the logs being saved in ./logs

    tensorboard --logdir ./logs

Once this is done, you can modify the average_models.py script to specify the paths to the model you want to load and then run it. 

    python3 average_models.py
