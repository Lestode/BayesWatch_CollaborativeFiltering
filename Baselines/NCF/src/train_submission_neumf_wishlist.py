import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from itertools import cycle

from data_loader import DataManager, RatingsDataset, WishlistBPRDataset
from models import GMF, MLP, NeuMF
from losses import BPRWishlistPlusMSELoss
from utils import set_seed

# --- Global Config ---
SEED = 42
DATA_DIR = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "./submission/NeuMF_BPR_MSE"

BPR_WEIGHTS = [0.1, 0.2, 0.3, 0.4]

HYPERPARAMETERS = {
    "model_dim": 64,
    "pretrain_epochs": 25, 
    "pretrain_lr_gmf": 0.00015,
    "pretrain_lr_mlp": 0.00001,
    "pretrain_batch_size": 256,
    "lr": 0.0002, 
    "batch_size": 128,
    "epochs": 80,
    "num_negative_samples_bpr": 30
}

def pretrain_component(component_type, full_train_df, n_users_global, n_items_global, 
                                 model_dim, pretrain_batch_size, pretrain_epochs, pretrain_lr, 
                                 device, bpr_weight,
                                 wishlist_df_for_pretrain, user_all_interactions_for_pretrain, 
                                 num_negative_samples_bpr_for_pretrain):
    """Pretrains GMF or MLP component for NeuMF on the full training data using BPRWishlistPlusMSELoss."""
    print(f"Pretraining {component_type} for NeuMF with model_dim={model_dim}, epochs={pretrain_epochs}, bpr_weight={bpr_weight}...")
    
    # mse dataloader
    users_tensor = torch.tensor(full_train_df["user_id"].values, dtype=torch.long)
    items_tensor = torch.tensor(full_train_df["item_id"].values, dtype=torch.long)
    ratings_tensor = torch.tensor(full_train_df["rating"].values, dtype=torch.float32)

    pretrain_mse_dataset = RatingsDataset(users_tensor, items_tensor, ratings_tensor)
    pretrain_mse_loader = DataLoader(pretrain_mse_dataset, batch_size=pretrain_batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)

    # bpr dataloader for pretraining
    pretrain_bpr_loader = None
    if wishlist_df_for_pretrain is not None and not wishlist_df_for_pretrain.empty and n_items_global > 0:
        pretrain_bpr_dataset = WishlistBPRDataset(
            wishlist_df=wishlist_df_for_pretrain,
            user_all_interactions=user_all_interactions_for_pretrain,
            n_items_global=n_items_global,
            num_negative_samples=num_negative_samples_bpr_for_pretrain
        )
        if len(pretrain_bpr_dataset) > 0:
            pretrain_bpr_loader = DataLoader(pretrain_bpr_dataset, batch_size=pretrain_batch_size, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
            print(f"  Created BPR DataLoader")
        else:
            print(f"  WishlistBPRDataset for pretraining {component_type} is empty")
    else:
        print(f"  Wishlist data for pretraining {component_type} is empty or n_items_global is 0")


    if component_type == 'gmf':
        component_model = GMF(n_users_global, n_items_global, model_dim)
    elif component_type == 'mlp':
        component_model = MLP(n_users_global, n_items_global, model_dim)
    else:
        raise ValueError("Invalid component type for pretraining")

    component_model.to(device)
    optimizer = optim.Adam(component_model.parameters(), lr=pretrain_lr)
    criterion = BPRWishlistPlusMSELoss(lambda_bpr=bpr_weight) 

    for epoch in range(pretrain_epochs):
        component_model.train()
        epoch_loss = 0.0
        num_batches = 0

        bpr_iter_pretrain = cycle(pretrain_bpr_loader) if pretrain_bpr_loader else None

        for mse_users, mse_items, mse_ratings in pretrain_mse_loader:
            mse_users, mse_items, mse_ratings = mse_users.to(device), mse_items.to(device), mse_ratings.to(device)
            
            optimizer.zero_grad()
            
            mse_predictions = component_model(mse_users, mse_items)
            
            bpr_users_for_loss_pretrain, bpr_pos_items_for_loss_pretrain, bpr_neg_items_batch_for_loss_pretrain = None, None, None
            if bpr_iter_pretrain:
                try:
                    # these are ids for the loss function
                    bpr_users_pt, bpr_pos_items_pt, bpr_neg_items_batch_pt = next(bpr_iter_pretrain)
                    bpr_users_pt, bpr_pos_items_pt, bpr_neg_items_batch_pt = bpr_users_pt.to(device), bpr_pos_items_pt.to(device), bpr_neg_items_batch_pt.to(device)

                    if bpr_users_pt.numel() > 0:
                        bpr_users_for_loss_pretrain = bpr_users_pt
                        bpr_pos_items_for_loss_pretrain = bpr_pos_items_pt
                        bpr_neg_items_batch_for_loss_pretrain = bpr_neg_items_batch_pt
                
                except StopIteration: 
                    pass 

            loss = criterion(mse_predictions, mse_ratings, 
                             component_model, # pass the model itself
                             bpr_users_for_loss_pretrain, 
                             bpr_pos_items_for_loss_pretrain, 
                             bpr_neg_items_batch_for_loss_pretrain)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"  Pretrain Epoch {epoch+1}/{pretrain_epochs} - {component_type} Loss: {avg_epoch_loss:.4f}")
    
    return component_model

def train_neumf_and_predict(hyperparameters: dict, output_dir: str):
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"--- Training NeuMF with BPR+MSE for submission ---")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data ---
    data_manager = DataManager(data_dir=DATA_DIR)
    train_ratings_df = data_manager.get_ratings_df()
    wishlist_df = data_manager.get_wishlist_df()
    user_all_interactions = data_manager.get_user_all_interactions()
    submission_df_to_predict = data_manager.get_submission_df()
    
    n_users_global = data_manager.get_num_users()
    n_items_global = data_manager.get_num_items()

    if train_ratings_df.empty or n_users_global == 0 or n_items_global == 0:
        print("No training data loaded for ratings. MSE part cannot proceed.")
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
    bpr_weight = hyperparameters.get('bpr_weight')
    num_negative_samples_bpr = hyperparameters.get('num_negative_samples_bpr')
    

    # --- Pretrain GMF and MLP Components ---
    gmf_pretrained = pretrain_component(
        'gmf', train_ratings_df, n_users_global, n_items_global,
        model_dim,
        pretrain_batch_size, pretrain_epochs, pretrain_lr_gmf, DEVICE, bpr_weight,
        wishlist_df, user_all_interactions, num_negative_samples_bpr
    )
    mlp_pretrained = pretrain_component(
        'mlp', train_ratings_df, n_users_global, n_items_global,
        model_dim,
        pretrain_batch_size, pretrain_epochs, pretrain_lr_mlp, DEVICE, bpr_weight,
        wishlist_df, user_all_interactions, num_negative_samples_bpr
    )

    model = NeuMF(n_users_global, n_items_global, model_dim)
    model.load_pretrained_weights(gmf_pretrained, mlp_pretrained, are_paths=False)
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    criterion = BPRWishlistPlusMSELoss(lambda_bpr=bpr_weight)

    train_users_t = torch.tensor(train_ratings_df["user_id"].values, dtype=torch.long)
    train_items_t = torch.tensor(train_ratings_df["item_id"].values, dtype=torch.long)
    train_ratings_t = torch.tensor(train_ratings_df["rating"].values, dtype=torch.float32)
    
    mse_dataset = RatingsDataset(train_users_t, train_items_t, train_ratings_t)
    mse_loader = DataLoader(mse_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)

    bpr_loader = None
    if wishlist_df is not None and not wishlist_df.empty and n_items_global > 0:
        bpr_dataset = WishlistBPRDataset(
            wishlist_df=wishlist_df,
            user_all_interactions=user_all_interactions,
            n_items_global=n_items_global,
            num_negative_samples=num_negative_samples_bpr
        )
        if len(bpr_dataset) > 0:
            bpr_loader = DataLoader(bpr_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
            print(f"Created BPR DataLoader with {len(bpr_dataset)} samples.")
        else:
            print("WishlistBPRDataset is empty.")
    else:
        print("Wishlist data is empty or n_items_global is 0.")

    # --- Training Loop ---
    print("Starting NeuMF model training with BPR+MSE loss...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        bpr_iter = cycle(bpr_loader) if bpr_loader else None

        for mse_users, mse_items, mse_ratings in mse_loader:
            mse_users, mse_items, mse_ratings = mse_users.to(DEVICE), mse_items.to(DEVICE), mse_ratings.to(DEVICE)
            
            optimizer.zero_grad()
            
            mse_predictions = model(mse_users, mse_items)
            
            bpr_users_for_loss, bpr_pos_items_for_loss, bpr_neg_items_batch_for_loss = None, None, None
            if bpr_iter:
                try:
                    # these are ids for the loss function
                    bpr_users, bpr_pos_items, bpr_neg_items_batch = next(bpr_iter)
                    bpr_users, bpr_pos_items, bpr_neg_items_batch = bpr_users.to(DEVICE), bpr_pos_items.to(DEVICE), bpr_neg_items_batch.to(DEVICE)

                    if bpr_users.numel() > 0:
                        bpr_users_for_loss = bpr_users
                        bpr_pos_items_for_loss = bpr_pos_items
                        bpr_neg_items_batch_for_loss = bpr_neg_items_batch
                
                except StopIteration:
                    pass

            loss = criterion(mse_predictions, mse_ratings, 
                             model, # pass the model itself
                             bpr_users_for_loss, 
                             bpr_pos_items_for_loss, 
                             bpr_neg_items_batch_for_loss)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in epoch {epoch+1}. Skipping batch.")
                print(f"MSE Preds: {mse_predictions.mean().item() if mse_predictions is not None else 'N/A'}, MSE Ratings: {mse_ratings.mean().item() if mse_ratings is not None else 'N/A'}")
                if bpr_users_for_loss is not None: print(f"BPR Pos: {bpr_pos_items_for_loss.mean().item()}")
                if bpr_neg_items_batch_for_loss is not None: print(f"BPR Neg: {bpr_neg_items_batch_for_loss.mean().item()}")
                continue


            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f}")

    # --- Save Model and Hyperparameters ---
    model_save_path = os.path.join(output_dir, "NeuMF_BPR_MSE_final_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved trained NeuMF model to {model_save_path}")

    hyperparams_save_path = os.path.join(output_dir, "NeuMF_BPR_MSE_hyperparams.json")
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
    
    submission_file_path = os.path.join(output_dir, "submission_NeuMF_BPR_MSE.csv")
    output_submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")

if __name__ == '__main__':
    for bpr_weight in BPR_WEIGHTS:
        hyperparameters_copy = HYPERPARAMETERS.copy()
        hyperparameters_copy['bpr_weight'] = bpr_weight
        train_neumf_and_predict(hyperparameters=hyperparameters_copy, output_dir=os.path.join(OUTPUT_DIR, str(bpr_weight)))