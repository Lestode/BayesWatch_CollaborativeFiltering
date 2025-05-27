import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import json
import itertools
import time

from data_loader import DataManager, RatingsDataset
from models import GMF, MLP, NeuMF
from losses import MSELoss
from utils import set_seed

# --- Configuration ---
SEED = 42

DATA_DIR = "./data"
RESULTS_DIR = "./cv_results"
N_EPOCHS = 80
N_PRETRAIN_EPOCHS = 25
N_FOLDS = 5

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set model config here
# model_config = {"name": "GMF", "type": "gmf"}
# model_config = {"name": "MLP", "type": "mlp"}
model_config = {"name": "NeuMF", "type": "neumf"}


# Set grid search params here (depends on model config)
MODEL_DIM_VALUES = [8, 16, 32, 64]
LEARNING_RATES = [0.0001, 0.0002, 0.0003]
BATCH_SIZES = [128, 256]

# for neumf model
# fixed pretrain params found through grid search on plain GMF and MLP models
PRETRAIN_LR_GMF = 0.00015
PRETRAIN_LR_MLP = 0.00001
PRETRAIN_BATCH_SIZE = 256


def train_eval_model_fold(model, train_loader, val_loader, optimizer, criterion, n_epochs, model_name_for_log="Model"):
    """Trains and evaluates a model for one fold."""
    best_val_loss = float('inf')
    
    model.to(DEVICE)

    for epoch in range(n_epochs):
        model.train()
        train_loss_epoch = 0.0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        
        avg_train_loss = train_loss_epoch / len(train_loader)

        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                val_loss_epoch += loss.item()
        
        avg_val_loss = val_loss_epoch / len(val_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs} - {model_name_for_log}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss

def pretrain_component(component_type, train_df_fold, n_users_global, n_items_global, model_dim, device):
    """Pretrains GMF or MLP component for NeuMF."""
    print(f"Pretraining {component_type} with model_dim={model_dim} for {N_PRETRAIN_EPOCHS} epochs...")
    
    users_tensor = torch.tensor(train_df_fold["user_id"].values, dtype=torch.long)
    items_tensor = torch.tensor(train_df_fold["item_id"].values, dtype=torch.long)
    ratings_tensor = torch.tensor(train_df_fold["rating"].values, dtype=torch.float32)
    
    pretrain_dataset = RatingsDataset(users_tensor, items_tensor, ratings_tensor)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True if device.type == 'cuda' else False)


    if component_type == 'gmf':
        component_model = GMF(n_users_global, n_items_global, model_dim)
        optimizer = optim.Adam(component_model.parameters(), lr=PRETRAIN_LR_GMF)
    elif component_type == 'mlp':
        component_model = MLP(n_users_global, n_items_global, model_dim)
        optimizer = optim.Adam(component_model.parameters(), lr=PRETRAIN_LR_MLP)
    else:
        raise ValueError("Invalid component type for pretraining")

    component_model.to(device)
    criterion = MSELoss()

    for epoch in range(N_PRETRAIN_EPOCHS):
        component_model.train()
        epoch_loss = 0
        for users, items, ratings in pretrain_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            predictions = component_model(users, items)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Pretrain Epoch {epoch+1}/{N_PRETRAIN_EPOCHS} - {component_type} Loss: {epoch_loss/len(pretrain_loader):.4f}")
    
    return component_model

# --- Main Cross-Validation Logic ---
# Finds the best hyperparameters for every model dimension
def run_cv():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    data_manager = DataManager(data_dir=DATA_DIR)
    ratings_df = data_manager.get_ratings_df()
    n_users_global = data_manager.get_num_users()
    n_items_global = data_manager.get_num_items()

    if ratings_df.empty or n_users_global == 0 or n_items_global == 0:
        print("No data loaded. Exiting cross-validation.")
        return

    criterion = MSELoss()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_name = model_config["name"]
    model_type = model_config["type"]
    print(f"\n--- Running Cross-Validation for {model_name} ---")
    
    model_results_dir = os.path.join(RESULTS_DIR, model_name.lower())
    os.makedirs(model_results_dir, exist_ok=True)

    for model_dim in MODEL_DIM_VALUES:
        print(f"  Model Dimension: {model_dim}")
        
        best_overall_val_loss_for_dim = float('inf')
        best_hyperparams_for_dim = {}
        avg_epoch_for_best_hyperparams_for_dim = 0

        current_hyperparam_grid = list(itertools.product(LEARNING_RATES, BATCH_SIZES))
        param_names = ['lr', 'batch_size']

        for hyperparams_tuple in current_hyperparam_grid:
            hyperparams = dict(zip(param_names, hyperparams_tuple))
            
            current_lr = hyperparams['lr']
            current_batch_size = hyperparams['batch_size']

            log_str = f"    Params: model_dim={model_dim}, lr={current_lr}, batch_size={current_batch_size}"
            print(log_str)

            fold_val_losses = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(ratings_df)):
                train_df_fold = ratings_df.iloc[train_idx]
                val_df_fold = ratings_df.iloc[val_idx]

                train_users_t = torch.tensor(train_df_fold["user_id"].values, dtype=torch.long)
                train_items_t = torch.tensor(train_df_fold["item_id"].values, dtype=torch.long)
                train_ratings_t = torch.tensor(train_df_fold["rating"].values, dtype=torch.float32)
                
                val_users_t = torch.tensor(val_df_fold["user_id"].values, dtype=torch.long)
                val_items_t = torch.tensor(val_df_fold["item_id"].values, dtype=torch.long)
                val_ratings_t = torch.tensor(val_df_fold["rating"].values, dtype=torch.float32)

                train_dataset = RatingsDataset(train_users_t, train_items_t, train_ratings_t)
                val_dataset = RatingsDataset(val_users_t, val_items_t, val_ratings_t)
                
                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
                val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)

                model = None
                optimizer = None

                if model_type == "gmf":
                    model = GMF(n_users_global, n_items_global, model_dim)
                    optimizer = optim.Adam(model.parameters(), lr=current_lr)
                elif model_type == "mlp":
                    model = MLP(n_users_global, n_items_global, model_dim)
                    optimizer = optim.Adam(model.parameters(), lr=current_lr)
                elif model_type == "neumf":
                    gmf_pretrained = pretrain_component('gmf', train_df_fold, n_users_global, n_items_global, model_dim, DEVICE)
                    mlp_pretrained = pretrain_component('mlp', train_df_fold, n_users_global, n_items_global, model_dim, DEVICE)
                    
                    model = NeuMF(n_users_global, n_items_global, model_dim)
                    model.load_pretrained_weights(gmf_pretrained, mlp_pretrained, are_paths=False)
                    optimizer = optim.SGD(model.parameters(), lr=current_lr)
                
                best_val_loss_fold = train_eval_model_fold(
                    model, train_loader, val_loader, optimizer, criterion, N_EPOCHS, f"{model_name} Fold {fold+1}"
                )
                fold_val_losses.append(best_val_loss_fold)

            avg_val_loss_for_params = np.mean(fold_val_losses)
            print(f"      Avg Val Loss for current params: {avg_val_loss_for_params:.4f}")

            if avg_val_loss_for_params < best_overall_val_loss_for_dim:
                best_overall_val_loss_for_dim = avg_val_loss_for_params
                best_hyperparams_for_dim = hyperparams.copy()
        
        output_data = {
            "model_dim": model_dim,
            "best_hyperparameters": best_hyperparams_for_dim,
            "best_average_val_loss": best_overall_val_loss_for_dim
        }
        
        output_filename = os.path.join(model_results_dir, f"dim_{model_dim}.json")
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"    Best results for model_dim {model_dim} saved to {output_filename}")
        print(f"    Best Hyperparams: {best_hyperparams_for_dim}, Avg Val Loss: {best_overall_val_loss_for_dim:.4f}, Avg Best Epoch: {avg_epoch_for_best_hyperparams_for_dim:.1f}")

if __name__ == '__main__':
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    
    start_time = time.time()
    run_cv()
    end_time = time.time()
    print(f"\nCross-validation finished in {(end_time - start_time)/60:.2f} minutes.")
