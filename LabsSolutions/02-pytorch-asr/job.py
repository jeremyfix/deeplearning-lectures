#!/usr/bin/env python3

import os
import subprocess


def makejob(commit_id, nruns, partition, walltime, augment, debug, params):
    paramsstr = " ".join([f"--{name} {value}" for name, value in params.items()])
    if augment:
        paramsstr += " --train_augment "
    if debug:
        paramsstr += " --debug "
    return f"""#!/bin/bash

#SBATCH --job-name=asr-ctc
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=0-{nruns-1}

current_dir=`pwd`

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
date

echo "Running on $(hostname)"

echo "Copying the source directory and data"
date
mkdir $TMPDIR/asr
rsync -r . $TMPDIR/asr/ --exclude 'logslurms' --exclude 'logs'

cd $TMPDIR/asr
git checkout {commit_id}

# Remove the index files to really test if regenerating them will work
rm *.idx

echo ""
echo "Virtual env"
date

python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate
python -m pip install -r requirements.txt

echo ""
echo "Training"
date

python main_ctc.py  --logname ctc_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} --baselogdir ${{current_dir}}/logs {paramsstr} train

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
result = int(
    subprocess.check_output("git status | grep 'modifi' | wc -l", shell=True).decode()
)
if result != 1:
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
debug = False
augment = True
submit_job(
    makejob(
        commit_id,
        1,
        "gpu_prod_long",
        "48:00:00",
        augment,
        debug,
        {
            "batch_size": 256,
            "num_epochs": 100,
            "base_lr": 0.001,
            "min_duration": 1.0,
            "max_duration": 5.0,
            "nlayers_rnn": 4,
            "nhidden_rnn": 1024,
            "weight_decay": 0.0,
            "dropout": 0.0,
            "datasetversion": "v15.0",
        },
    )
)
