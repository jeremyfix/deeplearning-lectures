I suggest you to train several models, you may need to adapt the parameters below


    for iter in $(seq 1 10); do \
    echo ">>>> Run $iter" && python3 train.py --dataset_dir ~/Datasets/FashionMNIST --use_gpu --num_workers 7 --normalize --model fancyCNN; done ;

Once this is done, you can modify the average_models.py script and run it. 

    python3 average_models.py
