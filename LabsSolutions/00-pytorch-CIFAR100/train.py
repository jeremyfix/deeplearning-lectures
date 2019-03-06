import torchvision.datasets

import argparse

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
        default=None
    )

    parser.add_argument(
            '--num_workers',
            type=int,
            default=1,
            help='The number of CPU threads used'
    )


    args = parser.parse_args()

    dataset = torchvision.datasets.CIFAR10(args.dataset_dir, train=True, transform=None, target_transform=None, download=False)
