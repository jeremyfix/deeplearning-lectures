---
title:  'Parametrizing a script with argparse'
author:
- Jeremy Fix
keywords: [argparse]
...

When you want to test a family of architectures, to be understood in a
large sense, covering both the model, its initialization, the loss,
optimization function, possibly preprocessing steps, ..., it is
certainly easier to write a single script that can take optional command
line arguments to parameterize it.

The
[argparse](https://docs.python.org/3/library/argparse.html#module-argparse)
python module is particularly well suited for defining a parameterized
script. A basic usage of argparse is to build an
[ArgumentParser](https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser)
object and to add arguments to it. Arguments can be optional or
mandatory, of type int, string, bool (a flag), etc..

The code below is an example of python script with some elements on how
to use it.

``` {.sourceCode .python}
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
```

And now some usage examples:

```console

mymachine:~:mylogin$ python3 argpase_ex.py -h
usage: argpase_ex.py [-h] [--normalize {None,minmax,std}] [--logdir LOGDIR]
                          --h1 H1 [--L2 L2 | --dropout DROPOUT]

optional arguments:
-h, --help            show this help message and exit
--normalize {None,minmax,std}
                      Which normalization to apply to the input data
--logdir LOGDIR       The directory in which to store the logs
--h1 H1               The size of the hidden layer
--L2 L2               Activate L2 regularization with the provided penalty
--dropout DROPOUT     Activate Dropout with the specified rate

mymachine:~:mylogin$ python3 argpase_ex.py
usage: argpase_ex.py [-h] [--normalize {None,minmax,std}] [--logdir LOGDIR]
                      --h1 H1 [--L2 L2 | --dropout DROPOUT]     
argpase_ex.py: error: the following arguments are required: --h1

mymachine:~:mylogin$ python3 argpase_ex.py --h1 10
Namespace(L2=None, dropout=None, h1=10, logdir='./logs', normalize='None')

mymachine:~:mylogin$ python3 argpase_ex.py --h1 10 --dropout 0.5 --normalize std
Namespace(L2=None, dropout=0.5, h1=10, logdir='./logs', normalize='std')
```
