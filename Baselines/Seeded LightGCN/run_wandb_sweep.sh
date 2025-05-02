#!/bin/bash

#SBATCH --output=logs/Novel.out
#SBATCH --error=logs/Novel.err
#SBATCH --job-name=Novel
#SBATCH -A cil_jobs
#SBATCH -G 1
#SBATCH -t 48:00:00

# wandb sweep wandb_sweep.yaml
wandb agent soufianebarrada0-eth-z-rich/cil/***

