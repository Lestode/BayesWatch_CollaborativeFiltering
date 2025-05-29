# Deep Matrix Factorization (DMF)

You'll find two files in this folder

- Basic implementation of the DMF paper: dmf.py
- Two stage training implementation: dmf_two_stage_training.py

## Run

To run those models, use the followings commands, while updating if necessary

- Run "python3 dmf_two_stage_training.py
  --hidden_dim1 64 \
   --hidden_dim2 16 \
   --embedding_dim 6 \
   --lr 1e-3 \
   --neg_ratio 7 \
   --batch_size 512 \
   --epochs_pretrain 25 \
   --implicit_score 1 \
   --hidden_dim_reg1 20 \
   --hidden_dim_reg2 20 \
   --lr_regression 6e-4 \
   --batch_size_regression 256 \
   --epochs_regression 10 \
   --device 'cpu' \
   --seed 2025 \
   --wandb False"
- Run "python3 dmf.py \
   --hidden_dim1 20 \
   --hidden_dim2 10 \
   --embedding_dim 20 \
   --lr 1e-3 \
   --neg_ratio 1 \
   --batch_size 512 \
   --epochs 25 \
   --model_path 'dmf_weights.pth' \
   --device 'cpu' \
   --loss "RMSE" \
   --seed 2025 \
   --use_wandb False"

## Requirements

- **Python** ≥ 3.7
- **NumPy** ≥ 1.18
- **PyTorch** ≥ 1.7
- **pandas** ≥ 1.1
- **wandb** ≥ 0.12 _(optional, only if you want Weights & Biases logging)_
