#!/usr/bin/env python3 # Added shebang for direct execution
import argparse
import os
import sys # For checking W&B environment variables

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import wandb # Deferred import

from helper import read_data_df, read_data_matrix, evaluate # Assumes these functions exist

# Global variable to hold the wandb module if loaded
wandb = None

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
        return sim.clamp(min=1e-6)

class InteractionDataset(Dataset):
    """
    Yields (user_idx, item_idx, rating) with negative sampling.
    """
    def __init__(self, matrix: torch.Tensor, neg_ratio: int = 1):
        super().__init__()
        self.matrix_shape = matrix.shape
        pos_indices = (matrix > 0).nonzero(as_tuple=True)
        
        self.interactions = []
        for r, c in zip(*pos_indices):
            self.interactions.append((r.item(), c.item(), matrix[r, c].item()))

        num_neg_samples_to_draw = len(self.interactions) * neg_ratio
        
        neg_count = 0
        max_attempts = num_neg_samples_to_draw * 10 
        attempts = 0
        # Store existing positive interactions for quick check during negative sampling
        positive_set = set(zip(pos_indices[0].tolist(), pos_indices[1].tolist()))

        while neg_count < num_neg_samples_to_draw and attempts < max_attempts:
            r_neg = torch.randint(0, self.matrix_shape[0], (1,)).item()
            c_neg = torch.randint(0, self.matrix_shape[1], (1,)).item()
            if (r_neg, c_neg) not in positive_set: # Check against the set
                self.interactions.append((r_neg, c_neg, 0.0))
                neg_count += 1
            attempts +=1
        if attempts == max_attempts and neg_count < num_neg_samples_to_draw:
            print(f"Warning: Could only sample {neg_count}/{num_neg_samples_to_draw} negative interactions.")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]


def train_dmf(matrix: torch.Tensor, config, device, wandb_enabled_for_run: bool):
    """
    Train the DMF model.
    config is either argparse.Namespace or wandb.config
    wandb_enabled_for_run indicates if W&B logging should be attempted.
    """
    matrix = matrix.to(device)
    num_users, num_items = matrix.size()

    model = DMFModel(
        num_user_inputs = num_items,
        num_item_inputs = num_users,
        hidden_dim1      = config.hidden_dim1,
        hidden_dim2      = config.hidden_dim2,
        embedding_dim    = config.embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataset = InteractionDataset(matrix, config.neg_ratio)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    max_rating = matrix.max().item()
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        for i_idxs, j_idxs, ratings in loader:
            i_idxs  = i_idxs.to(device, non_blocking=True)
            j_idxs  = j_idxs.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)

            rows = matrix[i_idxs]
            cols = matrix[:, j_idxs].t()

            preds = model(rows, cols)
            normalized_labels = (ratings.float() / max_rating).clamp(min=1e-6)
            if config.loss == "RMSE":
                loss = F.mse_loss(preds, normalized_labels)
            elif config.loss == "NORMALIZED CROSS ENTROPY":
                labels = (ratings.float() / max_rating).clamp(min=0.0, max=1.0) # BCE targets should be 0 or 1
                loss = F.binary_cross_entropy(preds, labels, reduction='mean') # mean is more common for BCE per batch
            else:
                raise ValueError(f"Unsupported loss function: {config.loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * ratings.size(0)

        epoch_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{config.epochs}, Loss: {epoch_loss:.4f}")
        
        # Conditional W&B logging
        if wandb_enabled_for_run and wandb and wandb.run:
            wandb.log({ 'train/epoch_loss': epoch_loss, 'epoch': epoch })

    return model


def main():
    global wandb # Allow assignment to the global wandb

    parser = argparse.ArgumentParser(description="Train DMF model, optionally with W&B Sweeps")
    parser.add_argument('--hidden_dim1', type=int, default=20)
    parser.add_argument('--hidden_dim2', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--neg_ratio', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model_path', type=str, default='dmf_weights.pth')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--wandb_project', type=str, default='cil_dmf_sweep')
    parser.add_argument('--wandb_entity', type=str, default='louis-barinka-eth-z-rich') # Replace with your entity
    parser.add_argument('--loss', type=str, default="RMSE", choices=["RMSE", "NORMALIZED CROSS ENTROPY"])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_wandb', type=bool, default=False, help="Enable W&B logging and integration")

    args = parser.parse_args()
    
    # This will be our single source of truth for hyperparameters
    # It starts as args, and becomes wandb.config if W&B is enabled and initialized.
    effective_config = args
    wandb_is_active = False

    # Determine if W&B should be used
    # (Explicitly requested via flag OR implicitly via sweep environment variables)
    should_try_wandb = args.use_wandb 

    if should_try_wandb:
        try:
            import wandb as wb_module # Import with an alias to assign to global `wandb`
            wandb = wb_module         # Make wandb accessible globally in this module
            
            # If an agent is running this script, wandb.init() might have already been called.
            # The W&B library handles this gracefully.
            # We pass `vars(args)` to `config` so that argparse defaults are available
            # to W&B, and sweep parameters can override them.
            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            
            if wandb_run:
                effective_config = wandb.config # Use wandb.config as the source of truth
                wandb_is_active = True
                print("W&B initialized successfully. Using W&B config.")
                # If WANDB_MODE=disabled, wandb.run will be a "DisabledRun" object.
                # wandb.log calls will be no-ops, which is desired.
                if wandb.run.disabled:
                    print("W&B is in disabled mode (e.g., WANDB_MODE=disabled). No data will be synced.")
            else:
                # This case (wandb.init returns None without error) is unlikely with modern wandb versions
                print("W&B initialization did not return a run object. Proceeding without W&B.")
        except ImportError:
            print("wandb library not found. Proceeding without W&B, even if requested.")
        except Exception as e: # Catch other wandb.init errors like auth errors
            print(f"Error initializing W&B: {e}. Proceeding without W&B.")
    else:
        print("W&B not requested and no W&B environment detected. Proceeding without W&B.")

    print("\nEffective Configuration:")
    if isinstance(effective_config, argparse.Namespace):
        for key, value in vars(effective_config).items():
            print(f"  {key}: {value}")
    else: # Should be wandb.config (which is dict-like)
        for key, value in effective_config.items():
            print(f"  {key}: {value}")
    print("-" * 30)

    # Set seed using the effective_config
    torch.manual_seed(effective_config.seed)
    np.random.seed(effective_config.seed)
    
    # Load data
    train_df, valid_df = read_data_df()
    train_mat = np.nan_to_num(read_data_matrix(train_df), nan=0.0)
    train_tensor = torch.from_numpy(train_mat).float()

    # Train using effective_config
    model = train_dmf(train_tensor, effective_config, torch.device(effective_config.device), wandb_is_active)

    # Determine model save path
    model_save_path = effective_config.model_path
    # If W&B is active, a run exists, and the model_path is the default, append run ID
    if wandb_is_active and wandb and wandb.run and not wandb.run.disabled and \
       effective_config.model_path == 'dmf_weights.pth':
        model_save_path = f"dmf_weights_{wandb.run.id}.pth"

    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


    # Define prediction function for evaluation
    def pred_fn_factory(train_tensor_local, model_local, max_rating_val_local, device_local):
        # Using local in names to avoid any potential capture issues if this factory were nested deeper
        def pred_fn(sids, pids):
            model_local.eval()
            s = torch.from_numpy(sids).long()
            p = torch.from_numpy(pids).long()
            preds_list = []
            batch_size_eval = 1024 
            with torch.no_grad():
                for i in range(0, len(s), batch_size_eval):
                    sb = s[i:i+batch_size_eval]
                    pb = p[i:i+batch_size_eval]
                    rows_eval = train_tensor_local[sb].to(device_local)
                    cols_eval = train_tensor_local[:, pb].t().to(device_local)
                    out = model_local(rows_eval, cols_eval).cpu().numpy()
                    preds_list.append(out)
            return np.concatenate(preds_list) * max_rating_val_local
        return pred_fn

    # Evaluate
    max_rating_val = train_mat.max().item()
    pred_fn = pred_fn_factory(train_tensor, model, max_rating_val, torch.device(effective_config.device))
    rmse = evaluate(valid_df, pred_fn)
    print(f"Validation RMSE: {rmse:.4f}")
    
    if wandb_is_active and wandb and wandb.run: # Log RMSE if W&B is active
        wandb.log({'val/rmse': rmse}) 

    if effective_config.loss == "NORMALIZED CROSS ENTROPY":
        valid_mat = np.nan_to_num(read_data_matrix(valid_df), nan=0.0)
        valid_tensor = torch.from_numpy(valid_mat).float().to(effective_config.device)
        
        # Create a new InteractionDataset for validation - ensure neg_ratio behavior is what you want for validation.
        # Often for validation, you evaluate on known positives and a fixed set of negatives, or all negatives.
        # Here, it will sample negatives similar to training if neg_ratio > 0.
        # If you want to evaluate NCE only on positive interactions, set neg_ratio=0 or modify dataset.
        valid_dataset = InteractionDataset(valid_tensor, effective_config.neg_ratio) # or neg_ratio=0 for only positives
        valid_loader  = DataLoader(valid_dataset, batch_size=effective_config.batch_size, shuffle=False)

        model.eval()
        total_nce = 0.0
        num_samples = 0
        with torch.no_grad():
            for u_idxs, i_idxs, ratings_val in valid_loader: # Renamed ratings to ratings_val
                u_idxs   = u_idxs.to(effective_config.device)
                i_idxs   = i_idxs.to(effective_config.device)
                ratings_val  = ratings_val.to(effective_config.device)

                rows = valid_tensor[u_idxs] # Use valid_tensor for fetching features
                cols = valid_tensor[:, i_idxs].t()
                preds = model(rows, cols)

                normalized_labels = (ratings_val.float() / max_rating_val).clamp(min=0.0, max=1.0) # BCE targets 0 or 1
                # Using reduction='sum' and then dividing by total samples gives a comparable NCE
                total_nce += F.binary_cross_entropy(preds, normalized_labels, reduction='sum').item()
                num_samples += ratings_val.size(0)

        avg_nce = total_nce / num_samples if num_samples > 0 else 0.0

        print(f"Validation Normalized Cross-Entropy: {avg_nce:.4f}")
        if wandb_is_active and wandb and wandb.run: # Log NCE if W&B is active
            wandb.log({'val/normalized_cross_entropy': avg_nce})

    if wandb_is_active and wandb and wandb.run:
        wandb.finish()
    
    print("Run completed.")

if __name__ == '__main__':
    # Ensure the script is executable (e.g., chmod +x your_script_name.py)
    main()