import os

# --- Project Root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAIN_RATINGS_PATH = os.path.join(DATA_DIR, 'train_ratings.csv')
TRAIN_WISHLIST_PATH = os.path.join(DATA_DIR, 'train_tbr.csv') 
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# --- Pre-trained Base Model Paths ---
BASE_MODELS_DIR = os.path.join(PROJECT_ROOT, 'trained_base_models')

# FunkSVD
FUNKSVD_MODEL_DIR = os.path.join(BASE_MODELS_DIR, 'funk_svd')
FUNKSVD_EXPLICIT_MODEL_PATH = os.path.join(FUNKSVD_MODEL_DIR, 'FunkSVD_model.pkl')

# LightGCN
LIGHTGCN_MODEL_DIR = os.path.join(BASE_MODELS_DIR, 'seeded_light_gcn')
LIGHTGCN_MODEL_PATH = os.path.join(LIGHTGCN_MODEL_DIR, 'lightgcn_final.pt')
SVD_SEED_FOR_LIGHTGCN_PATH = os.path.join(LIGHTGCN_MODEL_DIR, 'svd_embeddings.pt')

# NCF
NCF_MODEL_DIR = os.path.join(BASE_MODELS_DIR, 'ncf')
NEUMF_WISHLIST_MODEL_PATH = os.path.join(NCF_MODEL_DIR, 'NeuMF_wishlist.pth')
NCF_MAP_DIR = os.path.join(NCF_MODEL_DIR, 'maps')
NCF_USER_MAP_PATH = os.path.join(NCF_MAP_DIR, 'user_to_idx.json')
NCF_ITEM_MAP_PATH = os.path.join(NCF_MAP_DIR, 'item_to_idx.json')

# Weighted ALS
WEIGHTED_ALS_MODEL_DIR = os.path.join(BASE_MODELS_DIR, 'weighted_als')
WEIGHTED_ALS_MODEL_PATH = os.path.join(WEIGHTED_ALS_MODEL_DIR, 'weighted_als_model.pkl')

# NMF
NMF_MODEL_DIR = os.path.join(BASE_MODELS_DIR, 'nmf')
NMF_MODEL_PATH = os.path.join(NMF_MODEL_DIR, 'nmf_model.pkl')

# --- Final Submission Output ---
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, 'submissions')
FINAL_SUBMISSION_PATH = os.path.join(SUBMISSIONS_DIR, 'ensemble_submission.csv')

# --- Model Configurations (for loading wrappers) ---

# Seeded LightGCN necessary config for loading the model
LIGHTGCN_CONFIG = {
    'num_layers': 9,
    'dropout': 0.014265505942976877,
    'fine_tune_embed': True,
}

# NeuMF necessary config for loading the model
NEUMF_CONFIG = {
    'models_dim': 64
}

# --- Device Configuration (for PyTorch models) ---
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Weighted Average Ensemble Configuration ---

# PUBLIC SCORES in order from best to worst
# LightGCN: 0.83673
# FunkSVD: 0.83945
# NMF: 0.85131
# NeuMF: 0.85995
# WeightedALS: 0.86484

# Sum should be 1.0
AVERAGE_ENSEMBLE_WEIGHTS = {
    'LightGCN': 0.6,
    'FunkSVD': 0.3,
    'NMF': 0.1,
    'NeuMF': 0.0,
    'WeightedALS': 0.00,
}
