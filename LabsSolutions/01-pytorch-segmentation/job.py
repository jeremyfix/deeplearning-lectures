#!/usr/bin/env python3
"""
    This script belongs to the plankton classification sample code
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Script used to run experiments on a slurm cluster using the commit-id to guarantee the executed code version
"""

import os
import subprocess


def makejob(
    commit_id,
    nruns,
    partition,
    walltime,
    params,
):
    paramsstr = " ".join([f"--{name} {value}" for name, value in params.items()])
    return f"""#!/bin/bash

#SBATCH --job-name=semseg-{params['model']}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=0-{nruns-1}

current_dir=`pwd`

# Fix env variables as bashrc and profile are not loaded
export LOCAL=$HOME/.local
export PATH=$PATH:$LOCAL/bin

echo "Session " {params['model']}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
date

echo "Running on $(hostname)"
# env

echo "Copying the source directory and data"
date
mkdir $TMPDIR/lw_deeplearning_semantic
cd ../../
rsync -r . $TMPDIR/lw_deeplearning_semantic/ --exclude 'logs'  --exclude 'logslurms'

cd $TMPDIR/lw_deeplearning_semantic
cd LabsSolutions/03-pytorch-segmentation

ls 

git checkout {commit_id}

echo ""
echo "Conda virtual env"

export PATH=/opt/conda/bin:$PATH
source activate dl-lectures-segmentation


echo ""
echo "Training"
date

python main.py  --datadir /mounts/Datasets4/Stanford2D-3D-S/ {paramsstr} --logname {params['model']}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} --commit_id '{commit_id}' --logdir ${{current_dir}}/logs train

if [[ $? != 0 ]]; then
    exit -1
fi
date
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
result = int(subprocess.check_output("git status -s -uno | wc -l", shell=True).decode())
if result > 0:
    print(f"We found {result} modifications not staged or commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

# Ensure the log directory exists
os.system("mkdir -p logslurms")

# Launch the batch jobs
for model in ["UNet"]:
    submit_job(
        makejob(
            commit_id,
            1,
            "gpu_prod_long",
            "48:00:00",
            params={
                "model": model,
                "batch_size": 32,
                "weight_decay": 0.00,
                "nepochs": 100,
                "base_lr": 0.0001,
                "loss": "FocalLoss",
            },
        )
    )
