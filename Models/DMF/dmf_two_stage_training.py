#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Conditionally import wandb
try:
    import wandb
except ImportError:
    wandb = None # wandb will be None if not installed

from helper import read_data_df,read_data_matrix, evaluate# Assumes these functions exist

# Seed for reproducibility - will be set by config later

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

        def build_mlp(input_dim, hidden_dims_list, output_dim_mlp):
            layers = []
            current_dim = input_dim
            for h_dim in hidden_dims_list:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, output_dim_mlp)) # Layer to embedding_dim
            return nn.Sequential(*layers)

        self.user_net = build_mlp(num_user_inputs, hidden_dims_mlp_structure, embedding_dim)
        self.item_net = build_mlp(num_item_inputs, hidden_dims_mlp_structure, embedding_dim)

    def forward(self, user_features, item_features):
        u_embed = self.user_net(user_features)
        v_embed = self.item_net(item_features)
        sim = F.cosine_similarity(u_embed, v_embed, dim=-1)
        return sim.clamp(min=1e-6)

class DMFRegressionModel(nn.Module):
    """
    ### NEW ###: Wraps a pre-trained DMFModel and adds a regression head for rating prediction (Stage 2).
    """
    def __init__(self, dmf_model_pretrained: DMFModel, regression_head_hidden_dim1: int, regression_head_hidden_dim2: int):
        super().__init__()
        self.dmf_model_pretrained = dmf_model_pretrained

        for param in self.dmf_model_pretrained.parameters():
            param.requires_grad = False

        embedding_dim = self.dmf_model_pretrained.embedding_dim
        layers = []
        current_dim = 2 * embedding_dim

        layers.append(nn.Linear(current_dim, regression_head_hidden_dim1))
        layers.append(nn.ReLU())
        current_dim = regression_head_hidden_dim1

        layers.append(nn.Linear(current_dim, regression_head_hidden_dim2))
        layers.append(nn.ReLU())
        current_dim = regression_head_hidden_dim2

        layers.append(nn.Linear(current_dim, 1))
        self.regression_head = nn.Sequential(*layers)

    def forward(self, user_features, item_features):
        u_embed = self.dmf_model_pretrained.user_net(user_features)
        v_embed = self.dmf_model_pretrained.item_net(item_features)
        combined_embed = torch.cat((u_embed, v_embed), dim=-1)
        predicted_rating = self.regression_head(combined_embed)
        return predicted_rating.squeeze(-1)

class InteractionDataset(Dataset):
    def __init__(self, matrix: torch.Tensor, neg_ratio: int = 1):
        super().__init__()
        self.matrix_shape = matrix.shape
        pos_indices = (matrix > 0).nonzero(as_tuple=True)
        self.interactions = []
        for r, c in zip(*pos_indices):
            self.interactions.append((r.item(), c.item(), matrix[r, c].item()))

        num_pos_interactions = len(self.interactions)
        num_neg_samples_to_draw = num_pos_interactions * neg_ratio
        neg_count = 0
        max_attempts = num_neg_samples_to_draw * 20
        attempts = 0
        positive_set = set(zip(pos_indices[0].tolist(), pos_indices[1].tolist()))

        while neg_count < num_neg_samples_to_draw and attempts < max_attempts:
            r_neg = torch.randint(0, self.matrix_shape[0], (1,)).item()
            c_neg = torch.randint(0, self.matrix_shape[1], (1,)).item()
            if (r_neg, c_neg) not in positive_set:
                self.interactions.append((r_neg, c_neg, 0.0))
                neg_count += 1
            attempts +=1
        if attempts == max_attempts and neg_count < num_neg_samples_to_draw:
            print(f"Warning: Could only sample {neg_count}/{num_neg_samples_to_draw} negative interactions.")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]

class PositiveInteractionDataset(Dataset):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.matrix_shape = matrix.shape
        pos_indices = (matrix > 0).nonzero(as_tuple=True)
        self.interactions = []
        for r, c in zip(*pos_indices):
            self.interactions.append((r.item(), c.item(), matrix[r, c].item()))

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]

def train_dmf_pretraining(matrix: torch.Tensor, config, device, use_wandb: bool, wandb_run_obj):
    matrix = matrix.to(device)
    num_users, num_items = matrix.size()

    model = DMFModel(
        num_user_inputs=num_items,
        num_item_inputs=num_users,
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        embedding_dim=config.embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataset = InteractionDataset(matrix, config.neg_ratio)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    max_rating = matrix.max().item()
    if max_rating == 0: max_rating = 1.0

    print(f"Starting DMF Pre-training (Stage 1) for {config.epochs_pretrain} epochs.")
    model.train()
    for epoch in range(1, config.epochs_pretrain + 1):
        total_loss = 0.0
        for user_indices, item_indices, ratings in loader:
            user_indices = user_indices.to(device, non_blocking=True)
            item_indices = item_indices.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)

            user_features = matrix[user_indices]
            item_features = matrix[:, item_indices].t()
            preds_similarity = model(user_features, item_features)

            normalized_labels = (ratings.float() / max_rating).clamp(min=0.0, max=1.0)
            loss = F.binary_cross_entropy(preds_similarity, normalized_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ratings.size(0)

        epoch_loss = total_loss / len(dataset)
        print(f"Stage 1 - Epoch {epoch}/{config.epochs_pretrain}, Pre-train NCE Loss: {epoch_loss:.4f}")
        if use_wandb and wandb_run_obj and wandb: # Check wandb module exists
            wandb.log({'pretrain/epoch_nce_loss': epoch_loss, 'pretrain/epoch': epoch})

    print("Finished DMF Pre-training (Stage 1).")
    return model

def train_regression_head(dmf_model_pretrained: DMFModel, train_tensor: torch.Tensor, config, device, use_wandb: bool, wandb_run_obj):
    train_tensor = train_tensor.to(device)

    regression_model = DMFRegressionModel(
        dmf_model_pretrained,
        regression_head_hidden_dim1=config.hidden_dim_reg1,
        regression_head_hidden_dim2=config.hidden_dim_reg2
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, regression_model.parameters()),
        lr=config.lr_regression
    )

    dataset = PositiveInteractionDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size_regression, shuffle=True, num_workers=2, pin_memory=True)

    print(f"Starting Regression Head Training (Stage 2) for {config.epochs_regression} epochs.")
    regression_model.train()
    for epoch in range(1, config.epochs_regression + 1):
        total_mse_loss = 0.0
        for user_indices, item_indices, actual_ratings in loader:
            user_indices = user_indices.to(device, non_blocking=True)
            item_indices = item_indices.to(device, non_blocking=True)
            actual_ratings = actual_ratings.float().to(device, non_blocking=True)

            user_features = train_tensor[user_indices]
            item_features = train_tensor[:, item_indices].t()
            predicted_ratings = regression_model(user_features, item_features)
            loss = F.mse_loss(predicted_ratings, actual_ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse_loss += loss.item() * actual_ratings.size(0)

        epoch_mse_loss = total_mse_loss / len(dataset)
        epoch_rmse_loss = np.sqrt(epoch_mse_loss)
        print(f"Stage 2 - Epoch {epoch}/{config.epochs_regression}, MSE Loss: {epoch_mse_loss:.4f}, RMSE: {epoch_rmse_loss:.4f}")
        if use_wandb and wandb_run_obj and wandb: # Check wandb module exists
            wandb.log({
                'regression/epoch_mse_loss': epoch_mse_loss,
                'regression/epoch_rmse_loss': epoch_rmse_loss,
                'regression/epoch': epoch
            })

    print("Finished Regression Head Training (Stage 2).")
    return regression_model

def main():
    parser = argparse.ArgumentParser(description="Train 2-Stage DMF model with optional W&B Sweeps")

    # Stage 1 Hyperparameters
    parser.add_argument('--hidden_dim1', type=int, default=64)
    parser.add_argument('--hidden_dim2', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--neg_ratio', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs_pretrain', type=int, default=25)
    parser.add_argument('--implicit_score', type=int, default=1)

    # Stage 2 Hyperparameters
    parser.add_argument('--hidden_dim_reg1', type=int, default=20)
    parser.add_argument('--hidden_dim_reg2', type=int, default=20)
    parser.add_argument('--lr_regression', type=float, default=6e-4)
    parser.add_argument('--batch_size_regression', type=int, default=256)
    parser.add_argument('--epochs_regression', type=int, default=10)

    # General Config
    parser.add_argument('--model_path', type=str, default='dmf_regression_weights.pth')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--wandb_project', type=str, default='cil_dmf')
    parser.add_argument('--wandb_entity', type=str, default='louis-barinka-eth-z-rich')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--wandb', type=bool, default=False, help="Enable Weights & Biases logging and integration. E.g. --wandb True or --wandb False")

    args = parser.parse_args()
    current_run = None # Initialize current_run to None

    if args.wandb:
        if wandb is None:
            print("Wandb is enabled (--wandb True) but the wandb library is not installed. Please install it ('pip install wandb') to use wandb features.")
            print("Proceeding without wandb.")
            args.wandb = False # Disable wandb if library is not found
            config = args 
        else:
            # Initialize wandb. For sweeps, wandb.init() is often called by the agent,
            # or it picks up environment variables.
            # If running locally with --wandb True, project/entity are used.
            current_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
            
            # Merge argparse defaults/CLI args into wandb.config.
            # Sweep parameters in wandb.config (if any, from agent) will take precedence.
            # Create a dictionary from parsed args (CLI arguments and argparse defaults)
            cli_and_default_args_dict = vars(args)
            
            # Update this dictionary with parameters already in wandb.config (e.g., from a sweep).
            # This ensures sweep parameters override CLI/defaults.
            cli_and_default_args_dict.update(wandb.config) 
            
            # Now, update wandb.config with the merged dictionary.
            # This ensures wandb.config has all parameters, with sweep ones taking priority,
            # and new ones from argparse (not in sweep) are added.
            wandb.config.update(cli_and_default_args_dict, allow_val_change=True)
            config = wandb.config # Use wandb.config as the source of truth for hyperparameters
    else:
        config = args # Use parsed argparse arguments directly as config

    print("Effective Configuration:", config)

    # Set seed using the determined config object
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load data
    train_df, valid_df = read_data_df()
    train_mat = np.nan_to_num(read_data_matrix(train_df), nan=0.0)
    train_tensor = torch.from_numpy(train_mat).float()

    # --- Stage 1: Pre-training DMFModel ---
    dmf_model_pretrained = train_dmf_pretraining(
        train_tensor, config, torch.device(config.device),
        use_wandb=args.wandb, wandb_run_obj=current_run # Pass wandb status and run object
    )

    # --- Stage 2: Train Regression Head ---
    final_model = train_regression_head(
        dmf_model_pretrained.to(torch.device(config.device)),
        train_tensor,
        config,
        torch.device(config.device),
        use_wandb=args.wandb, wandb_run_obj=current_run # Pass wandb status and run object
    )

    # Save final model weights
    model_save_path = config.model_path
    # Check args.wandb (user's intent) and current_run (actual wandb session)
    if args.wandb and current_run and hasattr(current_run, 'id') and current_run.id:
         model_save_path = f"dmf_regression_weights_{current_run.id}.pth"

    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(final_model.state_dict(), model_save_path)
    print(f"Final model (DMFRegressionModel) saved to {model_save_path}")
    # if args.wandb and current_run and wandb:
    #     wandb.save(model_save_path) # Optionally save as W&B artifact

    def pred_fn_factory_regression(train_interaction_tensor, model_reg, device_eval):
        def pred_fn(sids, pids):
            model_reg.eval()
            s = torch.from_numpy(sids).long()
            p = torch.from_numpy(pids).long()
            preds_list = []
            batch_size_eval = 1024
            with torch.no_grad():
                for i in range(0, len(s), batch_size_eval):
                    sb = s[i:i+batch_size_eval]
                    pb = p[i:i+batch_size_eval]
                    rows_eval = train_interaction_tensor[sb].to(device_eval)
                    cols_eval = train_interaction_tensor[:, pb].t().to(device_eval)
                    out = model_reg(rows_eval, cols_eval).cpu().numpy()
                    preds_list.append(out)
            return np.concatenate(preds_list)
        return pred_fn

    # Evaluate final model
    # Corrected: train_tensor_explicit should be train_tensor as it holds the explicit ratings matrix
    pred_fn_reg = pred_fn_factory_regression(train_tensor, final_model, torch.device(config.device))
    rmse_final = evaluate(valid_df, pred_fn_reg)
    print(f"Final Validation RMSE: {rmse_final:.4f}")
    if args.wandb and current_run and wandb:
        wandb.log({'final_rmse': rmse_final})
    else:
        # If not using wandb, you might still want to see this in logs if running automated tests
        print(f"LOG_METRIC: {{'final_rmse': {rmse_final}}}")


    if config.epochs_pretrain > 0:
        valid_mat_np = np.nan_to_num(read_data_matrix(valid_df), nan=0.0)
        # Ensure valid_tensor is created on CPU first, then moved to device if needed
        valid_tensor_cpu = torch.from_numpy(valid_mat_np).float()
        
        max_rating_val = valid_mat_np.max().item()
        if max_rating_val == 0: max_rating_val = 1.0

        # Use InteractionDataset for validation NCE, with negative sampling
        # InteractionDataset expects tensor on CPU for initialization if it uses .tolist() or .item() extensively
        valid_dataset_nce = InteractionDataset(valid_tensor_cpu, config.neg_ratio)
        valid_loader_nce  = DataLoader(valid_dataset_nce, batch_size=config.batch_size, shuffle=False)

        dmf_model_pretrained.eval()
        total_val_nce = 0.0
        # Move valid_tensor to device for model input
        valid_tensor_device = valid_tensor_cpu.to(config.device)
        with torch.no_grad():
            for u_idxs, i_idxs, ratings_val in valid_loader_nce:
                u_idxs   = u_idxs.to(config.device)
                i_idxs   = i_idxs.to(config.device)
                ratings_val  = ratings_val.to(config.device)

                rows = valid_tensor_device[u_idxs]
                cols = valid_tensor_device[:, i_idxs].t()
                preds_sim_val = dmf_model_pretrained(rows, cols)

                normalized_labels_val = (ratings_val.float() / max_rating_val).clamp(min=0.0, max=1.0)
                total_val_nce += F.binary_cross_entropy(preds_sim_val, normalized_labels_val, reduction='sum').item()

        if len(valid_dataset_nce) > 0:
            avg_val_nce = total_val_nce / len(valid_dataset_nce) # This should be per example, not per batch sum.
                                                               # The loss was sum, so dividing by num_examples in dataset is correct.
            print(f"Validation NCE (after pre-training, Stage 1): {avg_val_nce:.4f}")
            if args.wandb and current_run and wandb:
                wandb.log({'val/pretrain_nce': avg_val_nce})
            else:
                print(f"LOG_METRIC: {{'val/pretrain_nce': {avg_val_nce}}}")
        else:
            print("Validation NCE: No positive samples in validation set for NCE calculation.")

    if args.wandb and current_run and wandb:
        wandb.finish()

if __name__ == '__main__':
    main()