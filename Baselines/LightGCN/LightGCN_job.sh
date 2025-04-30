#!/bin/bash
#SBATCH -A cil_jobs
#SBATCH -G 1
#SBATCH -t 20:00:00
#SBATCH --output=logs/LightGCN.out
#SBATCH --error=logs/LightGCN.err
#SBATCH --job-name=LightGCN

# module load cuda/12.6.0
# conda activate /cluster/courses/cil/envs/collaborative_filtering/

python LightGCN.py \
  --data_dir /cluster/courses/cil/collaborative_filtering/data \
  --output_dir lightgcn_output \
  --emb_dims   80 \
  --num_layers 30 \
  --lrs        2e-2 \
  --l1_reg     0 \
  --epochs     1000 \
  --val_split  0.2