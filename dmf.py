from typing import Tuple, Callable, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import wandb
import argparse

from helper import read_data_df, read_data_matrix, evaluate, make_submission # Assuming helper.py exists

# Seed need to be set for all experiments
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Your Model and Dataset classes remain the same ---
# class HybridLoss(nn.Module): ... # (Commented out in original, kept same)

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
        def build_mlp(dims: List[int]) -> nn.Sequential:
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        user_dims = [num_user_inputs] + hidden_dims + [embedding_dim]
        self.user_net = build_mlp(user_dims)

        item_dims = [num_item_inputs] + hidden_dims + [embedding_dim]
        self.item_net = build_mlp(item_dims)

    def forward(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        u = self.user_net(row)
        v = self.item_net(col)
        sim = F.cosine_similarity(u, v, dim=-1)
        return sim.clamp(min=1e-6)

class InteractionDataset(Dataset):
    """
    Dataset yielding (i, j, label) triplets for positive and sampled negative interactions.
    """
    def __init__(
        self,
        matrix: torch.Tensor,
        neg_ratio: int = 1
    ):
        pos = (matrix > 0).nonzero(as_tuple=False)
        neg = (matrix == 0).nonzero(as_tuple=False)
        pos_idx = pos.tolist()
        num_neg = len(pos_idx) * neg_ratio
        neg_idx = neg[torch.randperm(len(neg))[:num_neg]].tolist()
        self.interactions = [(i,j, matrix[i,j].item()) for i,j in pos_idx]
        self.interactions += [(i,j, 0.0) for i,j in neg_idx]

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, idx: int):
        i, j, r = self.interactions[idx]
        return i, j, r

def train_dmf(
    matrix: torch.Tensor,
    # --- Use hyperparameters from wandb.config ---
    config: wandb.Config, # Pass the config object
    device: torch.device = torch.device('cpu')
) -> DMFModel:
    """
    Train the DMF model using normalized cross-entropy loss (nce).
    """
    S, P = matrix.size()
    model = DMFModel(
        num_user_inputs = P,
        num_item_inputs = S,
        # --- Access hyperparameters from config ---
        hidden_dim1      = config.hidden_dim1,
        hidden_dim2      = config.hidden_dim2,
        embedding_dim    = config.embedding_dim
    ).to(device)

    # --- Important for Sweeps ---
    # Avoid saving/loading the same file across sweep runs unless you
    # incorporate the run ID into the filename. Usually disable during sweeps.
    # MODEL_PATH = f"dmf_weights_{wandb.run.id}.pth" # Example: Unique path per run
    MODEL_PATH = "dmf_weights.pth"
    # --- Disable loading for sweeps to ensure fair comparison ---
    # if os.path.exists(MODEL_PATH) and config.use_memorized_model: # Need to add use_memorized_model to config if needed
    #     try: # Add error handling in case file is corrupted or from incompatible run
    #       model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    #       print(" Loaded existing model")
    #       # Potentially return early ONLY if loading is the goal, not training
    #       # return model # Be careful with this logic during sweeps
    #     except Exception as e:
    #       print(f" Could not load model from {MODEL_PATH}: {e}")
    #       print(" Starting training from scratch.")

    # wandb.watch is automatically handled when wandb.init() is called before training
    # wandb.watch(model, log="all", log_freq=100) # Can keep if desired

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataset = InteractionDataset(matrix, config.neg_ratio)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True) # Added num_workers, pin_memory
    print(f"Using config: {config}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")

    max_rating = matrix.max().item()
    model.train() # Set model to training mode

    for epoch in range(1, config.epochs + 1):
        running_loss = 0.0
        for batch_idx, (i_idxs, j_idxs, ratings) in enumerate(loader):
            # Offload data loading if using num_workers
            # No need to manually print batch_idx, W&B logs steps
            i_idxs, j_idxs, ratings = i_idxs.to(device), j_idxs.to(device), ratings.to(device)

            # gather inputs (consider optimizing this if large)
            # Note: Slicing the matrix inside the loop can be slow.
            # It might be faster to pre-compute or use embedding layers if applicable.
            rows = matrix[i_idxs].to(device) # (batch_size, P)
            cols = matrix[:, j_idxs].t().to(device) # (batch_size, S)

            # forward
            preds = model(rows, cols)
            # normalize ratings to [0,1] *in float32*
            labels = ratings.float()  # Already on device
            norm_r = labels / max_rating
            loss = F.binary_cross_entropy(preds, norm_r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * ratings.size(0)

            # Log batch loss less frequently for cleaner plots
            if batch_idx % 100 == 0: # Log every 100 batches
                 # Log includes step (batch) and epoch automatically if using wandb.log
                wandb.log({"train/batch_loss": loss.item()}) # "epoch" and "batch_idx" are implicit/available

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{config.epochs}, Loss: {epoch_loss:.4f}")
        # log epoch-level loss - include epoch for clarity
        wandb.log({
            "train/epoch_loss": epoch_loss,
            "epoch": epoch
        })

    # --- Maybe disable saving during sweeps, or save with unique names ---
    # torch.save(model.state_dict(), MODEL_PATH)
    # print(f"💾 Model saved to {MODEL_PATH}") # Comment out for sweeps usually

    return model


# --- Main function to be called by the W&B agent ---
def run_sweep():
    wandb.login() # Ensure you're logged in - might not be needed if already logged in via CLI

    # Initialize W&B run *within this function*
    # The agent will automatically set the project, entity, and inject config
    run = wandb.init() # No arguments needed here when run by `wandb agent`

    # Access hyperparameters provided by the sweep agent via wandb.config
    # `wandb.config` is automatically populated by wandb.init() during a sweep
    cfg = wandb.config

    print("▶️ Starting run with config:", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (consider loading outside the function if it's slow and doesn't change)
    try:
        train_df, valid_df = read_data_df()
        train_mat = np.nan_to_num(read_data_matrix(train_df), nan=0.0)
        train_tensor = torch.from_numpy(train_mat).float() # Move to device later if needed
        max_rating_value = train_mat.max().item() # Get max rating
    except Exception as e:
        print(f"Error loading data: {e}")
        wandb.finish(exit_code=1) # Exit run if data loading fails
        return

    # Train the model using the hyperparameters from the sweep config
    model = train_dmf(
        matrix         = train_tensor, # Keep data on CPU for InteractionDataset slicing
        config         = cfg,          # Pass the config object
        device         = device
    )
    model.eval() # Set model to evaluation mode

    # Define prediction function for evaluation
    def pred_fn(sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        model.eval() # Ensure model is in eval mode
        # convert to torch
        s = torch.from_numpy(sids).long() # Keep indices on CPU
        p = torch.from_numpy(pids).long()

        # Batch processing for potentially large validation sets
        preds_list = []
        eval_batch_size = 1024 # Adjust as needed
        with torch.no_grad():
            for i in range(0, len(s), eval_batch_size):
                s_batch = s[i:i+eval_batch_size]
                p_batch = p[i:i+eval_batch_size]

                # grab the full row & column vectors for each pair
                # These need to be moved to the device for the model
                rows = train_tensor[s_batch].to(device)      # (batch, P)
                cols = train_tensor[:, p_batch].t().to(device) # (batch, S)

                preds_norm_batch = model(rows, cols).cpu().numpy() # (batch,)
                preds_list.append(preds_norm_batch)

        preds_norm = np.concatenate(preds_list)

        # rescale back to rating space
        # Ensure max_rating_value is accessible here (it is due to scope)
        return preds_norm * max_rating_value


    # Evaluate the model
    try:
        rmse = evaluate(valid_df, pred_fn)
        print(f"✅ Validation RMSE: {rmse:.4f}")
        # Log the metric that the sweep is optimizing
        wandb.log({"val/rmse": rmse})
    except Exception as e:
        print(f"Error during evaluation: {e}")
        wandb.log({"val/rmse": float('nan')}) # Log NaN if evaluation fails

    # Clean up the run
    wandb.finish()


run_sweep() 