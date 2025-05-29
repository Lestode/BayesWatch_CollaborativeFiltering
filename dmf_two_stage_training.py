#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
from helper import read_data_df, read_implicit_data_df,read_data_matrix, evaluate, build_interaction_matrix # Assumes these functions exist

# Seed for reproducibility - will be set by wandb.config later
# torch.manual_seed(42)
# np.random.seed(42)

class DMFModel(nn.Module):
    """
    Deep Matrix Factorization model with separate MLPs for users and items.
    ### MODIFIED ###: This base model is now primarily for Stage 1 (representation learning).
    Its output is cosine similarity.
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
        self.embedding_dim = embedding_dim
        hidden_dims_mlp_structure = [hidden_dim1, hidden_dim2] # Renamed for clarity within build_mlp scope

        ### MODIFIED ###: build_mlp slightly generalized for clarity.
        def build_mlp(input_dim, hidden_dims_list, output_dim_mlp):
            layers = []
            current_dim = input_dim
            for h_dim in hidden_dims_list:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, output_dim_mlp)) # Layer to embedding_dim
            return nn.Sequential(*layers)

        # User MLP: num_user_inputs -> hidden_dims -> embedding_dim
        self.user_net = build_mlp(num_user_inputs, hidden_dims_mlp_structure, embedding_dim)
        # Item MLP: num_item_inputs -> hidden_dims -> embedding_dim
        self.item_net = build_mlp(num_item_inputs, hidden_dims_mlp_structure, embedding_dim)

    # The model now expects the actual interaction vectors as input, not just indices.
    def forward(self, user_features, item_features):
        u_embed = self.user_net(user_features)
        v_embed = self.item_net(item_features)
        sim = F.cosine_similarity(u_embed, v_embed, dim=-1)
        return sim.clamp(min=1e-6) # Clamping for NCE/ranking loss during pre-training

class DMFRegressionModel(nn.Module):
    """
    ### NEW ###: Wraps a pre-trained DMFModel and adds a regression head for rating prediction (Stage 2).
    """
    def __init__(self, dmf_model_pretrained: DMFModel, regression_head_hidden_dim1: int, regression_head_hidden_dim2: int):
        super().__init__()
        self.dmf_model_pretrained = dmf_model_pretrained

        # Freeze the pre-trained user and item networks from Stage 1
        for param in self.dmf_model_pretrained.parameters():
            param.requires_grad = False

        # Embedding dimension from the pre-trained model
        embedding_dim = self.dmf_model_pretrained.embedding_dim

        # Regression head: Takes concatenated user and item embeddings
        # Input to regression head is 2 * embedding_dim (concatenated u_embed and v_embed)
        layers = []
        current_dim = 2 * embedding_dim

        # first hidden layer
        layers.append(nn.Linear(current_dim, regression_head_hidden_dim1))
        layers.append(nn.ReLU())
        current_dim = regression_head_hidden_dim1

        # second hidden layer
        layers.append(nn.Linear(current_dim, regression_head_hidden_dim2))
        layers.append(nn.ReLU())
        current_dim = regression_head_hidden_dim2

        # output
        layers.append(nn.Linear(current_dim, 1))


        self.regression_head = nn.Sequential(*layers)

    def forward(self, user_features, item_features):
        # Get embeddings from the pre-trained (and frozen) networks
        # No need to call dmf_model_pretrained.eval() here if it has no dropout/BN,
        # but it's good practice if it might be extended later.
        u_embed = self.dmf_model_pretrained.user_net(user_features)
        v_embed = self.dmf_model_pretrained.item_net(item_features)

        # Concatenate embeddings to feed into the regression head
        combined_embed = torch.cat((u_embed, v_embed), dim=-1)
        predicted_rating = self.regression_head(combined_embed)
        return predicted_rating.squeeze(-1) # Remove last dim to get (batch_size,) tensor of ratings

class InteractionDataset(Dataset):
    """
    Yields (user_idx, item_idx, rating) with negative sampling.
    Used for Stage 1 (ranking pre-training).
    ### MODIFIED ###: Minor fixes and small optimization. Largely original logic.
    """
    def __init__(self, matrix: torch.Tensor, neg_ratio: int = 1):
        super().__init__() ### FIX ###: Was missing parentheses in original.
        self.matrix_shape = matrix.shape
        pos_indices = (matrix > 0).nonzero(as_tuple=True)
        self.interactions = []
        for r, c in zip(*pos_indices):
            self.interactions.append((r.item(), c.item(), matrix[r, c].item()))

        num_pos_interactions = len(self.interactions) # Store for clarity
        num_neg_samples_to_draw = num_pos_interactions * neg_ratio

        neg_count = 0
        max_attempts = num_neg_samples_to_draw * 20 # Increased max attempts
        attempts = 0
        ### MODIFIED ###: Optimization for negative sampling check
        positive_set = set(zip(pos_indices[0].tolist(), pos_indices[1].tolist()))

        while neg_count < num_neg_samples_to_draw and attempts < max_attempts:
            r_neg = torch.randint(0, self.matrix_shape[0], (1,)).item()
            c_neg = torch.randint(0, self.matrix_shape[1], (1,)).item()
            # if matrix[r_neg, c_neg] == 0: # Original check, can be slow
            if (r_neg, c_neg) not in positive_set: # Faster check
                self.interactions.append((r_neg, c_neg, 0.0)) # Negative samples have a "rating" of 0
                neg_count += 1
            attempts +=1
        if attempts == max_attempts and neg_count < num_neg_samples_to_draw:
            print(f"Warning: Could only sample {neg_count}/{num_neg_samples_to_draw} negative interactions.")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]

class PositiveInteractionDataset(Dataset):
    """
    ### NEW ###: Yields (user_idx, item_idx, actual_rating) for positive interactions only.
    Used for Stage 2 (regression fine-tuning).
    """
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.matrix_shape = matrix.shape
        pos_indices = (matrix > 0).nonzero(as_tuple=True) # Use as_tuple=True for direct indexing
        self.interactions = []
        for r, c in zip(*pos_indices):
            self.interactions.append((r.item(), c.item(), matrix[r, c].item()))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        # Returns user_idx, item_idx, rating
        return self.interactions[idx]


### MODIFIED ###: This function is adapted from the original `train_dmf` for Stage 1 pre-training.
def train_dmf_pretraining(matrix: torch.Tensor, config, device):
    """
    Train the base DMFModel for representation learning (Stage 1).
    Uses a ranking-style loss (Normalized Cross Entropy / Binary Cross Entropy on similarities).
    """
    matrix = matrix.to(device)
    num_users, num_items = matrix.size()

    # Initialize the base DMFModel
    model = DMFModel(
        num_user_inputs=num_items,      # Each user is represented by their row of interactions
        num_item_inputs=num_users,      # Each item is represented by its column of interactions
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        embedding_dim=config.embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # Use InteractionDataset with negative sampling for pre-training
    dataset = InteractionDataset(matrix, config.neg_ratio)
    # Consider adding num_workers and pin_memory for Dataloader if not already present
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)


    max_rating = matrix.max().item()
    if max_rating == 0: max_rating = 1.0 # Avoid division by zero if matrix is all zeros (e.g. during dev)


    print(f"Starting DMF Pre-training (Stage 1) for {config.epochs_pretrain} epochs.")
    model.train()
    for epoch in range(1, config.epochs_pretrain + 1):
        total_loss = 0.0
        for user_indices, item_indices, ratings in loader:
            user_indices = user_indices.to(device, non_blocking=True)
            item_indices = item_indices.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)

            # ### MODIFIED ###: Inputs to the model are now the full interaction vectors (rows/cols)
            user_features = matrix[user_indices]        # Shape: (batch_size, num_items)
            item_features = matrix[:, item_indices].t() # Shape: (batch_size, num_users)

            preds_similarity = model(user_features, item_features) # Output is similarity

            # ### MODIFIED ###: Loss function changed for pre-training (ranking objective)
            # Normalized Cross Entropy: Treat positive interactions as "1" (or normalized rating)
            # and negative interactions (rating=0) as "0".
            normalized_labels = (ratings.float() / max_rating).clamp(min=0.0, max=1.0)
            # Binary Cross Entropy is suitable here for (similarity, 0/1-like target)
            loss = F.binary_cross_entropy(preds_similarity, normalized_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * ratings.size(0) # Multiply by batch size for correct average

        epoch_loss = total_loss / len(dataset)
        print(f"Stage 1 - Epoch {epoch}/{config.epochs_pretrain}, Pre-train NCE Loss: {epoch_loss:.4f}")
        wandb.log({'pretrain/epoch_nce_loss': epoch_loss, 'pretrain/epoch': epoch})

    print("Finished DMF Pre-training (Stage 1).")
    return model # Return the pre-trained base DMF model

def train_regression_head(dmf_model_pretrained: DMFModel, train_tensor: torch.Tensor, config, device):
    """
    ### NEW ###: Train the regression head on top of the frozen pre-trained DMF model (Stage 2).
    Uses RMSE (MSELoss) on actual ratings.
    """
    train_tensor = train_tensor.to(device) # Ensure tensor is on the correct device

    # Initialize the DMFRegressionModel, which wraps the pre-trained DMFModel
    regression_model = DMFRegressionModel(
        dmf_model_pretrained, # This model's user_net and item_net are frozen
        regression_head_hidden_dim1=config.hidden_dim_reg1,
        regression_head_hidden_dim2=config.hidden_dim_reg2
    ).to(device)

    # Optimizer for ONLY the regression head parameters
    # The dmf_model_pretrained parameters have requires_grad=False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, regression_model.parameters()),
        lr=config.lr_regression
    )

    # Use PositiveInteractionDataset for training with actual ratings (no negative sampling)
    dataset = PositiveInteractionDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size_regression, shuffle=True, num_workers=2, pin_memory=True)

    print(f"Starting Regression Head Training (Stage 2) for {config.epochs_regression} epochs.")
    regression_model.train() # Set the regression_model (and its submodules) to train mode
    for epoch in range(1, config.epochs_regression + 1):
        total_mse_loss = 0.0
        for user_indices, item_indices, actual_ratings in loader:
            user_indices = user_indices.to(device, non_blocking=True)
            item_indices = item_indices.to(device, non_blocking=True)
            actual_ratings = actual_ratings.float().to(device, non_blocking=True) # Target ratings

            # Inputs are still the full interaction vectors
            user_features = train_tensor[user_indices]
            item_features = train_tensor[:, item_indices].t()

            # The regression_model directly predicts ratings
            predicted_ratings = regression_model(user_features, item_features)

            # Loss is Mean Squared Error for rating prediction
            loss = F.mse_loss(predicted_ratings, actual_ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse_loss += loss.item() * actual_ratings.size(0)

        epoch_mse_loss = total_mse_loss / len(dataset)
        epoch_rmse_loss = np.sqrt(epoch_mse_loss) # RMSE is sqrt(MSE)
        print(f"Stage 2 - Epoch {epoch}/{config.epochs_regression}, MSE Loss: {epoch_mse_loss:.4f}, RMSE: {epoch_rmse_loss:.4f}")
        wandb.log({
            'regression/epoch_mse_loss': epoch_mse_loss,
            'regression/epoch_rmse_loss': epoch_rmse_loss, # Log training RMSE
            'regression/epoch': epoch
        })

    print("Finished Regression Head Training (Stage 2).")
    return regression_model # Return the full model with the trained regression head


def main():
    ### MODIFIED ###: Argument parser updated for two-stage hyperparameters.
    parser = argparse.ArgumentParser(description="Train 2-Stage DMF model with W&B Sweeps")

    # Stage 1 (Pre-training DMFModel) Hyperparameters
    parser.add_argument('--hidden_dim1', type=int, default=64, help="DMF MLP hidden dim 1 for pre-training")
    parser.add_argument('--hidden_dim2', type=int, default=16, help="DMF MLP hidden dim 2 for pre-training")
    parser.add_argument('--embedding_dim', type=int, default=6, help="DMF embedding dimension (pre-training)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for pre-training (Stage 1)")
    parser.add_argument('--neg_ratio', type=int, default=7, help="Negative sampling ratio for pre-training")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for pre-training (Stage 1)")
    parser.add_argument('--epochs_pretrain', type=int, default=25, help="Epochs for pre-training (Stage 1)")
    parser.add_argument('--implicit_score', type=int, default=1)

    # Stage 2 (Regression Head) Hyperparameters
    ### NEW ###: Hyperparameters specific to the regression head.
    parser.add_argument('--hidden_dim_reg1', type=int, default=20, help="Regression head hidden dim1")
    parser.add_argument('--hidden_dim_reg2', type=int, default=10, help="Regression head hidden dim2")
    parser.add_argument('--lr_regression', type=float, default=1e-3, help="Learning rate for regression head (Stage 2)")
    parser.add_argument('--batch_size_regression', type=int, default=256, help="Batch size for regression head training (Stage 2)")
    parser.add_argument('--epochs_regression', type=int, default=10, help="Epochs for regression head training (Stage 2)")

    # General Config (largely same as original, some defaults might change)
    parser.add_argument('--model_path', type=str, default='dmf_regression_weights.pth', help="Path to save final model")
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--wandb_project', type=str, default='cil_dmf') # Potentially new project
    parser.add_argument('--wandb_entity', type=str, default='louis-barinka-eth-z-rich') # Replace with your W&B entity
    parser.add_argument('--seed', type=int, default=42)
    # Removed --loss argument as Stage 1 uses NCE-like, Stage 2 uses RMSE.
    parser.add_argument('--wandb', type=bool, default=False)

    # Initialize wandb
    run = wandb.init() # For sweeps, agent calls init. For local, it picks up from env or default.
    # Merge argparse defaults into wandb.config if not already set by sweep
    # This ensures all expected keys are in wandb.config
    args = parser.parse_args()   # ← no empty list
    config_dict = vars(args)

    config_dict.update(wandb.config) # wandb.config (from sweep) overrides defaults
    wandb.config.update(config_dict, allow_val_change=True) # Update wandb.config with merged
    config = wandb.config # Use this config object throughout


    print("W&B Config (after merging defaults and parsing):", config)

    # Set seed using W&B config
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.device == 'cuda' and torch.cuda.is_available(): # Check cuda availability
        torch.cuda.manual_seed_all(config.seed)


    # Load data (same as original)
    train_df, valid_df = read_data_df()
    train_mat = np.nan_to_num(read_data_matrix(train_df), nan=0.0)
    train_tensor = torch.from_numpy(train_mat).float()

    # --- Stage 1: Pre-training DMFModel ---
    ### NEW ###: Call the pre-training function.
    dmf_model_pretrained = train_dmf_pretraining(train_tensor, config, torch.device(config.device))

    # --- Stage 2: Train Regression Head ---
    ### NEW ###: Call the regression head training function.
    # Ensure the pre-trained model is on the correct device before passing to regression trainer
    final_model = train_regression_head(
        dmf_model_pretrained.to(torch.device(config.device)),
        train_tensor, # train_tensor is already loaded, will be moved to device inside train_regression_head
        config,
        torch.device(config.device)
    )

    # Save final model weights (which is the DMFRegressionModel)
    model_save_path = config.model_path
    if hasattr(wandb.run, 'id') and wandb.run.id: # If it's a W&B run, make path unique
         model_save_path = f"dmf_regression_weights_{wandb.run.id}.pth"

    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir): # Ensure directory exists
        os.makedirs(save_dir)
    torch.save(final_model.state_dict(), model_save_path)
    print(f"Final model (DMFRegressionModel) saved to {model_save_path}")
    # wandb.save(model_save_path) # Optionally save as W&B artifact

    # ### MODIFIED ###: Prediction function factory for the final DMFRegressionModel.
    def pred_fn_factory_regression(train_interaction_tensor, model_reg, device_eval):
        def pred_fn(sids, pids): # user_ids, item_ids
            model_reg.eval() # Set to evaluation mode
            s = torch.from_numpy(sids).long() # user indices
            p = torch.from_numpy(pids).long() # item indices
            preds_list = []
            batch_size_eval = 1024 # Can be different from training batch_size
            with torch.no_grad():
                for i in range(0, len(s), batch_size_eval):
                    sb = s[i:i+batch_size_eval]
                    pb = p[i:i+batch_size_eval]

                    # Get full user/item feature vectors from the training interaction matrix
                    rows_eval = train_interaction_tensor[sb].to(device_eval)
                    cols_eval = train_interaction_tensor[:, pb].t().to(device_eval)

                    # The DMFRegressionModel directly predicts ratings
                    out = model_reg(rows_eval, cols_eval).cpu().numpy()
                    preds_list.append(out)
            # ### MODIFIED ###: No max_rating scaling needed as model predicts ratings directly.
            return np.concatenate(preds_list)
        return pred_fn

    # Evaluate final model
    # Ensure train_tensor is available for pred_fn_factory_regression
    pred_fn_reg = pred_fn_factory_regression(train_tensor_explicit, final_model, torch.device(config.device))
    # The evaluate function is assumed to take (validation_dataframe, prediction_function)
    # and return RMSE.
    rmse_final = evaluate(valid_df, pred_fn_reg)
    print(f"Final Validation RMSE: {rmse_final:.4f}")
    wandb.log({'final_rmse': rmse_final}) # This is the primary metric for sweeps optimization

    ### NEW ###: Optional: Log NCE on validation set using the pre-trained model (before regression head)
    # This can give insight into how well Stage 1 performed on its own task.
    if config.epochs_pretrain > 0: # Only if pretraining was actually done
        valid_mat_np = np.nan_to_num(read_data_matrix(valid_df), nan=0.0)
        valid_tensor = torch.from_numpy(valid_mat_np).float().to(config.device)
        max_rating_val = valid_mat_np.max().item()
        if max_rating_val == 0: max_rating_val = 1.0 # Avoid div by zero

        # Use InteractionDataset for validation NCE, with negative sampling
        valid_dataset_nce = InteractionDataset(valid_tensor, config.neg_ratio)
        valid_loader_nce  = DataLoader(valid_dataset_nce, batch_size=config.batch_size, shuffle=False)

        dmf_model_pretrained.eval() # Set pre-trained model to eval mode
        total_val_nce = 0.0
        with torch.no_grad():
            for u_idxs, i_idxs, ratings_val in valid_loader_nce:
                u_idxs   = u_idxs.to(config.device)
                i_idxs   = i_idxs.to(config.device)
                ratings_val  = ratings_val.to(config.device)

                # Get features for validation
                rows = valid_tensor[u_idxs]
                cols = valid_tensor[:, i_idxs].t()
                preds_sim_val = dmf_model_pretrained(rows, cols) # Similarity output

                normalized_labels_val = (ratings_val.float() / max_rating_val).clamp(min=0.0, max=1.0)
                total_val_nce += F.binary_cross_entropy(preds_sim_val, normalized_labels_val, reduction='sum').item()

        if len(valid_dataset_nce) > 0:
            avg_val_nce = total_val_nce / len(valid_dataset_nce)
            print(f"Validation NCE (after pre-training, Stage 1): {avg_val_nce:.4f}")
            wandb.log({'val/pretrain_nce': avg_val_nce})
        else:
            print("Validation NCE: No positive samples in validation set for NCE calculation.")


    wandb.finish()

if __name__ == '__main__':
    # Make sure to make the script executable: chmod +x your_script_name.py
    main()