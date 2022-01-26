#!/usr/bin/env python3

import os
import subprocess

def makejob(commit_id,
            nruns,
            partition,
            walltime,
            normalize,
            augment,
            params):
    paramsstr = " ".join([f"--{name} {value}" for name, value in params.items() ])
    if(normalize):
        paramsstr += " --normalize "
    if augment:
        paramsstr += " --data_augment "
    return f"""#!/bin/bash 

#SBATCH --job-name=lightning-{params['model']}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=0-{nruns-1}

current_dir=`pwd`

echo "Session " {params['model']}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
date

echo "Running on $(hostname)"

echo "Copying the source directory and data"
date

rsync -r . $TMPDIR --exclude 'logslurms' --exclude 'logs'

cd $TMPDIR/
git checkout {commit_id}

echo ""
echo "Virtual env setting"

virtualenv -p python3 venv
source venv/bin/activate
python -m pip install -r requirements_cuda.txt

echo ""
echo "Training"
date

python train.py {paramsstr} 

if [[ $? != 0 ]]; then
    exit -1
fi

date

"""

def submit_job(job):
	with open('job.sbatch', 'w') as fp:
		fp.write(job)
	os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
result = int(subprocess.check_output("git status | grep 'modifi' | wc -l", shell=True).decode())
if result != 1:
    print(f"We found {result} modifications not staged or commited")
    raise RuntimeError("You must stage and commit every modification before submission ")

commit_id = subprocess.check_output("git log --pretty=format:'%H' -n 1", shell=True).decode()

# Ensure the log directory exists
os.system("mkdir -p logslurms")

# Launch the batch jobs
data_augment = True
normalize = True
submit_job(makejob(commit_id, 2, 'gpu_prod_long', "1:00:00",
                   normalize, data_augment,
                   {'model': 'linear',
                    'weight_decay': 0.00,
                   }))
