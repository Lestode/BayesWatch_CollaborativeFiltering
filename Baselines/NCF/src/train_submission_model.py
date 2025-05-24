import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

from data_loader import DataManager, RatingsDataset
from models import GMF, MLP, NeuMF
from losses import MSELoss
from utils import set_seed

# --- Global Config ---
SEED = 42
DATA_DIR = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration ---
MODEL_TYPE = "NeuMF" # "GMF", "MLP", "NeuMF"
OUTPUT_DIR = "./submission/NeuMF"

HYPERPARAMETERS = {
    "model_dim": 64,
    "lr": 0.0002,
    "batch_size": 128,
    "epochs": 80,
    "pretrain_batch_size": 256,
    "pretrain_lr_gmf": 0.00015,
    "pretrain_lr_mlp": 0.00001,
    "pretrain_epochs": 25,
}

def pretrain_component(component_type, full_train_df, n_users_global, n_items_global, 
                             model_dim, pretrain_batch_size, pretrain_epochs, pretrain_lr, device):
    """Pretrains GMF or MLP component on the full training data."""
    print(f"Pretraining final {component_type} with model_dim={model_dim} for {pretrain_epochs} epochs...")
    
    users_tensor = torch.tensor(full_train_df["user_id"].values, dtype=torch.long)
    items_tensor = torch.tensor(full_train_df["item_id"].values, dtype=torch.long)
    ratings_tensor = torch.tensor(full_train_df["rating"].values, dtype=torch.float32)
    
    pretrain_dataset = RatingsDataset(users_tensor, items_tensor, ratings_tensor)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=pretrain_batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)

    if component_type == 'gmf':
        component_model = GMF(n_users_global, n_items_global, model_dim)
    elif component_type == 'mlp':
        component_model = MLP(n_users_global, n_items_global, model_dim)
    else:
        raise ValueError("Invalid component type for pretraining")

    component_model.to(device)
    optimizer = optim.Adam(component_model.parameters(), lr=pretrain_lr)
    criterion = MSELoss()

    for epoch in range(pretrain_epochs):
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
        print(f"  Pretrain Epoch {epoch+1}/{pretrain_epochs} - {component_type} Loss: {epoch_loss/len(pretrain_loader):.4f}")
    
    return component_model

def train_and_predict(model_type: str, hyperparameters: dict, output_dir: str):
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"--- Training {model_type} for submission ---")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data ---
    data_manager = DataManager(data_dir=DATA_DIR)
    train_ratings_df = data_manager.get_ratings_df()
    submission_df_to_predict = data_manager.get_submission_df()
    n_users_global = data_manager.get_num_users()
    n_items_global = data_manager.get_num_items()

    if train_ratings_df.empty or n_users_global == 0 or n_items_global == 0:
        print("No training data loaded. Cannot proceed.")
        return
    
    global_average_rating = train_ratings_df['rating'].mean()
    print(f"Global average rating from training data: {global_average_rating:.4f}")


    # --- Hyperparameters ---
    model_dim = hyperparameters.get('model_dim')
    lr = hyperparameters.get('lr')
    batch_size = hyperparameters.get('batch_size')
    epochs = hyperparameters.get('epochs')
    pretrain_epochs = hyperparameters.get('pretrain_epochs')
    pretrain_lr_gmf = hyperparameters.get('pretrain_lr_gmf')
    pretrain_lr_mlp = hyperparameters.get('pretrain_lr_mlp')
    pretrain_batch_size = hyperparameters.get('pretrain_batch_size')

    print({
        lr, batch_size, epochs, pretrain_epochs
    })
    
    model = None
    optimizer = None

    if model_type == "GMF":
        model = GMF(n_users_global, n_items_global, model_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif model_type == "MLP":
        model = MLP(n_users_global, n_items_global, model_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif model_type == "NeuMF":
        # --- Pretrain GMF and MLP Components ---
        gmf_pretrained = pretrain_component('gmf', train_ratings_df, n_users_global, n_items_global, 
                                                  model_dim, pretrain_batch_size, pretrain_epochs, pretrain_lr_gmf, DEVICE)
        mlp_pretrained = pretrain_component('mlp', train_ratings_df, n_users_global, n_items_global,
                                                  model_dim, pretrain_batch_size, pretrain_epochs, pretrain_lr_mlp, DEVICE)
        
        model = NeuMF(n_users_global, n_items_global, model_dim)
        model.load_pretrained_weights(gmf_pretrained, mlp_pretrained, are_paths=False)
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(DEVICE)
    criterion = MSELoss()

    train_users_t = torch.tensor(train_ratings_df["user_id"].values, dtype=torch.long)
    train_items_t = torch.tensor(train_ratings_df["item_id"].values, dtype=torch.long)
    train_ratings_t = torch.tensor(train_ratings_df["rating"].values, dtype=torch.float32)
    
    full_train_dataset = RatingsDataset(train_users_t, train_items_t, train_ratings_t)
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)

    # --- Training Loop ---
    print("Starting final model training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for users, items, ratings in full_train_loader:
            users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(full_train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}")

    # --- Save Model and Hyperparameters ---
    model_save_path = os.path.join(output_dir, f"{model_type}_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved trained model to {model_save_path}")

    hyperparams_save_path = os.path.join(output_dir, f"{model_type}_hyperparams.json")
    with open(hyperparams_save_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Saved hyperparameters to {hyperparams_save_path}")

    # --- Prediction for Submission ---
    if submission_df_to_predict.empty:
        print("No submission data to predict.")
        return

    print("Generating predictions for submission file...")
    model.eval()
    
    submission_users_t = torch.tensor(submission_df_to_predict['user_id'].astype(int).values, dtype=torch.long)
    submission_items_t = torch.tensor(submission_df_to_predict['item_id'].astype(int).values, dtype=torch.long)
    
    predict_dataset = RatingsDataset(submission_users_t, submission_items_t, ratings_tensor=None)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)

    all_predictions = []
    with torch.no_grad():
        for users, items in predict_loader:
            users, items = users.to(DEVICE), items.to(DEVICE)
            predictions = model(users, items)
            all_predictions.extend(predictions.cpu().numpy().tolist())
            
    submission_df_to_predict['rating'] = all_predictions

    output_submission_df = submission_df_to_predict[['sid_pid', 'rating']]
    
    submission_file_path = os.path.join(output_dir, f"submission_{model_type}.csv")
    output_submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")

if __name__ == '__main__':
    train_and_predict(model_type=MODEL_TYPE, hyperparameters=HYPERPARAMETERS, output_dir=OUTPUT_DIR)
