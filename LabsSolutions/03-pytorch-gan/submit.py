#!/usr/bin/python

import os
import sys


def makejob(dataset, nruns):
    baselog = os.path.abspath("./logs")
    return f"""#!/bin/bash

#SBATCH --job-name=tpGAN
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

export PATH=$PATH:~/.local/bin

current_dir=`pwd`

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/tpGAN
rsync -r --exclude "logs" --exclude "logslurms" --exclude venv . $TMPDIR/tpGAN
cd $TMPDIR/tpGAN

echo "Setting up the virtual environment"
python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate
python -m pip install -r requirements.txt

echo "Training"
python main.py --dataset {dataset} --dataset_root /mounts/Datasets4 --logdir {baselog}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} --num_epochs 400 train

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [1, 2]:
    print(f"Usage : {sys.argv[0]} <nruns|1>")
    sys.exit(-1)

if len(sys.argv) == 1:
    nruns = 1
else:
    nruns = int(sys.argv[1])

# Launch the batch jobs
for dataset in ["MNIST", "EMNIST", "FashionMNIST", "SVHN", "CelebA"]:
    submit_job(makejob(dataset, nruns))
