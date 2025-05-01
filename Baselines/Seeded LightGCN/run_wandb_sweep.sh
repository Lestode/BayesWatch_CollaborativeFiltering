#!/bin/bash

#SBATCH --output=logs/Novel.out
#SBATCH --error=logs/Novel.err
#SBATCH --job-name=Novel
#SBATCH -A cil_jobs
#SBATCH -G 1
#SBATCH -t 48:00:00

#module load cuda/12.6.0
#conda activate /cluster/courses/cil/envs/collaborative_filtering/

# wandb sweep wandb_sweep.yaml
wandb agent soufianebarrada0-eth-z-rich/cil/o3cy1770

