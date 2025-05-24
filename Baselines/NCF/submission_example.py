import torch
import pandas as pd
import os
import numpy as np

from src.data_loader import DataManager, RatingsDataset
from src.models import GMF, MLP, NeuMF
from torch.utils.data import DataLoader

# --- Configuration. TODO: Set before running the submission generation script ---
MODEL_TYPE = "NeuMF"  # Options: "MLP", "NeuMF", "GMF" (no GMF model was created for submission)

MODEL_PATH = "./models/NeuMF_wishlist.pth" # Path to the trained model file (.pth)
DATA_DIR = "./data"  # Directory containing 'train_ratings.csv' and 'sample_submission.csv'
OUTPUT_CSV_PATH = "./submission_NeuMF_wishlist.csv"  # Full path for the output submission CSV

MODEL_DIM = 64  # Dimensionality of embeddings (must match the trained model)
# Note: for the trained models in the save_models directory, the model_dim is 16 for MLP and 64 for NeuMF and wishlist-augmented NeuMF

BATCH_SIZE_PREDICT = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_RATING = 3.0 # Default rating for unmappable user/item pairs if training data is unavailable. If it is used, something is wrong.

def generate_submission(model_type: str, model_path: str, data_dir: str,
                        output_csv_path: str, model_dim: int,
                        batch_size: int, device: torch.device):
    """
    Generates predictions for the submission file using a trained model.
    """
    print(f"--- Generating submission using {model_type} model ---")
    print(f"Model path: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Model dimension: {model_dim}")
    print(f"Using device: {device}")

    # --- Load Data ---
    data_manager = DataManager(data_dir=data_dir)
    submission_df_original = data_manager.get_submission_df()
    n_users_global = data_manager.get_num_users()
    n_items_global = data_manager.get_num_items()
    
    train_ratings_df = data_manager.get_ratings_df()

    if submission_df_original is None or submission_df_original.empty:
        print("No submission data loaded. Cannot proceed.")
        return

    if n_users_global == 0 or n_items_global == 0:
        print(f"Number of users or items is zero. Cannot proceed.")
        return

    if train_ratings_df is None or train_ratings_df.empty:
        print(f"Training ratings data is empty or not available. Cannot proceed.")
        return

    # --- Initialize Model ---
    model = None
    if model_type == "GMF":
        model = GMF(n_users_global, n_items_global, model_dim)
    elif model_type == "MLP":
        model = MLP(n_users_global, n_items_global, model_dim)
    elif model_type == "NeuMF":
        model = NeuMF(n_users_global, n_items_global, model_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Load Model Weights ---
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model weights from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR: Could not load model weights: {e}")
        return
        
    model.to(device)
    model.eval()

    # --- Prepare for Prediction ---
    submission_df_original['rating'] = DEFAULT_RATING

    predictable_mask = submission_df_original['user_id'].notna() & submission_df_original['item_id'].notna()
    predictable_df = submission_df_original[predictable_mask].copy()

    if predictable_df.empty:
        print("No predictable user-item pairs in the submission file (all user_ids/item_ids are NaN).")
    else:
        print(f"Found {len(predictable_df)} pairs with known user/item IDs for model prediction.")
        predictable_df['user_id'] = predictable_df['user_id'].astype(int)
        predictable_df['item_id'] = predictable_df['item_id'].astype(int)

        submission_users_t = torch.tensor(predictable_df['user_id'].values, dtype=torch.long)
        submission_items_t = torch.tensor(predictable_df['item_id'].values, dtype=torch.long)

        predict_dataset = RatingsDataset(submission_users_t, submission_items_t, ratings_tensor=None)
        predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 

        all_model_predictions = []
        print("Generating model predictions...")
        with torch.no_grad():
            for users, items in predict_loader:
                users, items = users.to(device), items.to(device)
                predictions = model(users, items)
                all_model_predictions.extend(predictions.cpu().numpy().tolist())
        
        # Assign model predictions to the corresponding rows
        submission_df_original.loc[predictable_mask, 'rating'] = all_model_predictions
        print(f"Assigned {len(all_model_predictions)} model predictions.")
        
    num_default_predictions = len(submission_df_original) - len(predictable_df)
    if num_default_predictions > 0:
        print(f"Used default rating for {num_default_predictions} pairs due to unknown user/item IDs.")

    # --- Save Submission File ---
    output_submission_df = submission_df_original[['sid_pid', 'rating']]
    
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    output_submission_df.to_csv(output_csv_path, index=False)
    print(f"Submission file saved to {output_csv_path}")

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: MODEL_PATH '{MODEL_PATH}' does not exist.")
    elif not os.path.isdir(DATA_DIR):
         print(f"ERROR: DATA_DIR '{DATA_DIR}' does not exist or is not a directory.")
    else:
        generate_submission(
            model_type=MODEL_TYPE,
            model_path=MODEL_PATH,
            data_dir=DATA_DIR,
            output_csv_path=OUTPUT_CSV_PATH,
            model_dim=MODEL_DIM,
            batch_size=BATCH_SIZE_PREDICT,
            device=DEVICE
        )
