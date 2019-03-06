import argparse

import data
import models

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Whether to use GPU'
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Where to store the downloaded dataset',
        required=True
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--model',
        choices=['linear', 'cnn'],
        action='store',
        required=True
    )

    args = parser.parse_args()

    valid_ratio = 0.2
    batch_size  = 128
    num_workers = args.num_workers
    dataset_dir = args.dataset_dir
    train_augment_transform = []

    input_dim = (3, 32, 32)
    num_classes = 100

    # Data loading
    train_loader, valid_loader = data.load_data(valid_ratio,
                                                batch_size,
                                                num_workers,
                                                dataset_dir,
                                                train_augment_transform)
    # Model definition
    model = models.build_model(args.model, input_dim, num_classes)

    # Loss function

    # Callbacks

    # Training loop


