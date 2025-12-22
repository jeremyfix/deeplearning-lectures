import argparse

parser = argparse.ArgumentParser()

# Argument definition
parser.add_argument(
    '--normalize',
    choices=['None', 'minmax', 'std'],
    default='None',
    help='Which normalization to apply to the input data',
    action='store'
)
parser.add_argument(
    '--logdir',
    type=str,
    default="./logs",
    help='The directory in which to store the logs'
)
parser.add_argument(
    '--h1',
    type=int,
    required=True,
    help='The size of the hidden layer'
)
group_reg = parser.add_mutually_exclusive_group()
group_reg.add_argument(
    '--L2',
    type=float,
    help='Activate L2 regularization with the provided penalty'
)
group_reg.add_argument(
    '--dropout',
    type=float,
    help='Activate Dropout with the specified rate'
)

# Actual parsing
args = parser.parse_args()

print(args)
