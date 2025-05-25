#!/usr/bin/env python3 # Added shebang for direct execution
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

from helper import read_data_df, read_data_matrix, evaluate # Assumes these functions exist

# Seed for reproducibility - good to have, but wandb can also manage this
# torch.manual_seed(42) # Can be set by wandb.config if swept
# np.random.seed(42)  # Can be set by wandb.config if swept

class DMFModel(nn.Module):
    """
    Deep Matrix Factorization model with separate MLPs for users and items.
    """
    def __init__(
        self,
        num_user_inputs: int,
        num_item_inputs: int,
        hidden_dim1: int,
        hidden_dim2: int,
        embedding_dim: int
    ):
        super().__init__()
        hidden_dims = [hidden_dim1, hidden_dim2]

        def build_mlp(dims):
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2: # No ReLU after the last hidden layer before embedding
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        user_dims = [num_user_inputs] + hidden_dims + [embedding_dim]
        self.user_net = build_mlp(user_dims)

        item_dims = [num_item_inputs] + hidden_dims + [embedding_dim]
        self.item_net = build_mlp(item_dims)

    def forward(self, row, col):
        u = self.user_net(row)
        v = self.item_net(col)
        sim = F.cosine_similarity(u, v, dim=-1)
        return sim.clamp(min=1e-6) # Clamping similarity seems okay, usually done for predicted ratings

class InteractionDataset(Dataset):
    """
    Yields (user_idx, item_idx, rating) with negative sampling.
    """
    def __init__(self, matrix: torch.Tensor, neg_ratio: int = 1):
        super().__init__()
        self.matrix_shape = matrix.shape
        pos_indices = (matrix > 0).nonzero(as_tuple=True) # Use as_tuple=True for direct indexing
        
        self.interactions = []
        for r, c in zip(*pos_indices):
            self.interactions.append((r.item(), c.item(), matrix[r, c].item()))

        num_neg_samples_to_draw = len(self.interactions) * neg_ratio
        
        # Efficient negative sampling:
        # Keep track of potential negative samples and draw without replacement
        # This can be slow for very sparse matrices if we regenerate all negatives each time
        # A simpler approach is to randomly sample (r,c) pairs and check if matrix[r,c]==0
        neg_count = 0
        max_attempts = num_neg_samples_to_draw * 10 # Avoid infinite loop
        attempts = 0
        while neg_count < num_neg_samples_to_draw and attempts < max_attempts:
            r_neg = torch.randint(0, self.matrix_shape[0], (1,)).item()
            c_neg = torch.randint(0, self.matrix_shape[1], (1,)).item()
            if matrix[r_neg, c_neg] == 0:
                self.interactions.append((r_neg, c_neg, 0.0))
                neg_count += 1
            attempts +=1
        if attempts == max_attempts and neg_count < num_neg_samples_to_draw:
            print(f"Warning: Could only sample {neg_count}/{num_neg_samples_to_draw} negative interactions.")


    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]


def train_dmf(matrix: torch.Tensor, config, device): # config is now wandb.config
    """
    Train the DMF model.
    """
    matrix = matrix.to(device)
    num_users, num_items = matrix.size()

    model = DMFModel(
        num_user_inputs = num_items,
        num_item_inputs = num_users,
        hidden_dim1      = config.hidden_dim1, # Use wandb.config
        hidden_dim2      = config.hidden_dim2, # Use wandb.config
        embedding_dim    = config.embedding_dim  # Use wandb.config
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) # Use wandb.config
    dataset = InteractionDataset(matrix, config.neg_ratio) # Use wandb.config
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True) # Use wandb.config

    max_rating = matrix.max().item()
    model.train()
    for epoch in range(1, config.epochs + 1): # Use wandb.config
        total_loss = 0.0
        for i_idxs, j_idxs, ratings in loader:
            i_idxs  = i_idxs.to(device, non_blocking=True)
            j_idxs  = j_idxs.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)

            rows = matrix[i_idxs]
            cols = matrix[:, j_idxs].t()

            preds = model(rows, cols)
            normalized_labels = (ratings.float() / max_rating).clamp(min=1e-6) # Ensure labels are normalized like preds
            if config.loss == "RMSE":
                loss = F.mse_loss(preds, normalized_labels) # MSE is fine for normalized values
            elif config.loss == "NORMALIZED CROSS ENTROPY":
                labels = (ratings.float() / max_rating).clamp(min=0.0, max=1.0)
                loss = F.binary_cross_entropy(preds, labels, reduction='sum')
            else:
                raise ValueError(f"Unsupported loss function: {config.loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * ratings.size(0)

        epoch_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{config.epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({ 'train/epoch_loss': epoch_loss, 'epoch': epoch })

    return model


def main():
    # No need for argparse here if all config comes from wandb.config
    # However, it's good practice to keep it for standalone runs or to set defaults
    # that wandb can override.
    parser = argparse.ArgumentParser(description="Train DMF model with W&B Sweeps")
    # Add ALL parameters you want to sweep or configure to argparse
    parser.add_argument('--hidden_dim1', type=int, default=20)
    parser.add_argument('--hidden_dim2', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--neg_ratio', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model_path', type=str, default='dmf_weights.pth') # Might want to make this run-specific
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--wandb_project', type=str, default='cil_dmf_sweep') # Potentially new project for sweeps
    parser.add_argument('--wandb_entity', type=str, default='louis-barinka-eth-z-rich')
    parser.add_argument('--loss', type=str) #Either RMSE or 
    parser.add_argument('--seed', type=int, default=42) # Add seed if you want to sweep it or control it

    # Initialize wandb
    # For sweeps, wandb.init() will be called by the agent with parameters.
    # If running standalone, it will use defaults or cmd line args.
    run = wandb.init(project=parser.parse_args().wandb_project, entity=parser.parse_args().wandb_entity) # project/entity from args

    # Access all hyperparameters through wandb.config
    # wandb.config will be populated by sweep agent or by argparse defaults if not in a sweep
    # It's a good practice to merge argparse defaults into wandb.config if not already there
    # when wandb.init() is called without args (e.g. by an agent)
    # If an agent calls this script, it will pass --key=value, which argparse handles.
    # Then wandb.init() will automatically pick up these args for its config.
    
    # If you want to ensure args are used to populate wandb.config if not set by sweep:
    temp_args = parser.parse_args() # Parse once to get project/entity for init
    wandb.config.update({k: v for k, v in vars(temp_args).items() if k not in wandb.config}, allow_val_change=True)


    print("W&B Config:", wandb.config)

    # Set seed
    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    
    # Load data
    train_df, valid_df = read_data_df()
    train_mat = np.nan_to_num(read_data_matrix(train_df), nan=0.0)
    train_tensor = torch.from_numpy(train_mat).float()

    # Train using wandb.config
    model = train_dmf(train_tensor, wandb.config, torch.device(wandb.config.device))

    # Save model weights (optional, consider if you need all models from a sweep)
    # You might want to name it using the run ID
    model_save_path = f"dmf_weights_{wandb.run.id}.pth"
    if wandb.config.model_path != 'dmf_weights.pth': # If user specified a different default
        model_save_path = wandb.config.model_path

    # Create directory if it doesn't exist
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    # wandb.save(model_save_path) # You can also save it as a W&B artifact

    # Define prediction function for evaluation
    def pred_fn_factory(train_tensor, model, max_rating_val, device):
        def pred_fn(sids, pids):
            model.eval()
            s = torch.from_numpy(sids).long()
            p = torch.from_numpy(pids).long()
            preds_list = [] # Renamed to avoid conflict
            batch_size_eval = 1024 # Can be different from training batch_size
            with torch.no_grad():
                for i in range(0, len(s), batch_size_eval):
                    sb = s[i:i+batch_size_eval]
                    pb = p[i:i+batch_size_eval]
                    rows_eval = train_tensor[sb].to(device) # Renamed to avoid conflict
                    cols_eval = train_tensor[:, pb].t().to(device) # Renamed to avoid conflict
                    out = model(rows_eval, cols_eval).cpu().numpy()
                    preds_list.append(out)
            return np.concatenate(preds_list) * max_rating_val
        return pred_fn

    # Evaluate
    max_rating_val = train_mat.max().item()
    pred_fn = pred_fn_factory(train_tensor, model, max_rating_val, torch.device(wandb.config.device))
    rmse = evaluate(valid_df, pred_fn)
    print(f"Validation RMSE: {rmse:.4f}")
    wandb.log({'rmse': rmse}) # This is crucial for the sweep to optimize!

    if wandb.config.loss == "NORMALIZED CROSS ENTROPY":
        # 1) build validation tensor
        valid_mat = np.nan_to_num(read_data_matrix(valid_df), nan=0.0)
        valid_tensor = torch.from_numpy(valid_mat).float().to(wandb.config.device)

        # 2) wrap in dataset & loader
        valid_dataset = InteractionDataset(valid_tensor, wandb.config.neg_ratio)
        valid_loader  = DataLoader(valid_dataset, batch_size=wandb.config.batch_size, shuffle=False)

        # 3) accumulate BCE loss over all val samples
        model.eval()
        total_nce = 0.0
        with torch.no_grad():
            for u_idxs, i_idxs, ratings in valid_loader:
                u_idxs   = u_idxs.to(wandb.config.device)
                i_idxs   = i_idxs.to(wandb.config.device)
                ratings  = ratings.to(wandb.config.device)

                rows = valid_tensor[u_idxs]
                cols = valid_tensor[:, i_idxs].t()
                preds = model(rows, cols)

                normalized_labels = (ratings.float() / max_rating_val).clamp(min=1e-6)
                total_nce += F.binary_cross_entropy(preds, normalized_labels, reduction='sum').item()

        avg_nce = total_nce / len(valid_dataset)

        print(f"Validation normalized cross‐entropy: {avg_nce:.4f}")
        wandb.log({'val/normalized_cross_entropy': avg_nce})


    wandb.finish()

if __name__ == '__main__':
    # Make sure to make the script executable: chmod +x dmf.py
    main()