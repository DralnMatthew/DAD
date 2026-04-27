#!/bin/bash -l
#SBATCH --job-name=ces
#SBATCH --output=out/%x_%j_%a.txt
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --account=project_462001370

export EBU_USER_PREFIX=/scratch/project_462001370/Informed-BED/env2/EasyBuild

export WANDB_API_KEY=$(cat ~/.wandb_secret)

module --force purge

module use $EBU_USER_PREFIX/modules/container

module load PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827

cd /scratch/project_462001370/DAD

python ces.py wandb.use_wandb=True eval_batch_size=20 