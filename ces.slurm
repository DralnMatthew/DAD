#!/bin/bash -l

#SBATCH --time=100:00:00
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --output=out/%x_%j_%a.txt
#SBATCH --job-name=location
#SBATCH --array=1
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g
#SBATCH --mail-type=END
#SBATCH --mail-user=zhanghu.zhao@aalto.fi

export PYTHONUNBUFFERED=1

cd /scratch/work/zhaoz5/DAD

python ces.py wandb.use_wandb=True eval_batch_size=20