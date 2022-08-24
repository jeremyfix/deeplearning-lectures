#!/bin/bash

#SBATCH --job-name=SemSegEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=1:00:00
#SBATCH --output=logslurms/slurm-conda-%j.out
#SBATCH --error=logslurms/slurm-conda-%j.err

# Load the conda module
export PATH=/opt/conda/bin:$PATH

# Create conda environment 
# and prevent the use of system site-package
# that will interfer with the conda env
export PYTHONNOUSERSITE=1
conda create --name dl-lectures-segmentation python=3.9 --force
source activate dl-lectures-segmentation
pip install -r requirements.txt
