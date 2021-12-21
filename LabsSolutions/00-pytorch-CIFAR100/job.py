#!/usr/bin/env python3

import os
import copy

def makejob(nruns, partition, walltime,
            params, data_augment):
    paramsstr = " ".join([f"--{name} {value}" for name, value in params.items() ])
    if data_augment:
        paramsstr += ' --data_augment '
    return f"""#!/bin/bash 

#SBATCH --job-name=cifar-{params['model']}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=0-{nruns}

date

python3 train.py --use_gpu --dataset_dir $TMPDIR {paramsstr}

date

"""

def submit_job(job):
	with open('job.sbatch', 'w') as fp:
		fp.write(job)
	os.system("sbatch job.sbatch")


# Ensure the log directory exists
os.system("mkdir -p logslurms")

models = ['linear', 'cnn', 'resnet18']
normalization = ['None', 'channel_meanstd']

augmentation_params =[
    [
        True, 
        {
            'weight_decay': 0.01,
            'dropout': 0.5
        }
    ],
    [
        False,
        {
            'weight_decay': 0.0,
            'dropout': 0.0
        }
    ]
]


def joindir(d1, d2):
    d11 = copy.copy(d1)
    for k, v in d2.items():
        d11[k] = v
    return d11

# Launch the batch jobs
for m in models:
    for n in normalization:
        for augparams in augmentation_params:
            submit_job(makejob(2, 'gpu_prod', "2:00:00",
                               joindir({'model': m, 'normalization': n}, augparams[1]),
                               augparams[0]))
