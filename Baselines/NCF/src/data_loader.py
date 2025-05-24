import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import random

class DataManager:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.ratings_df = None
        self.wishlist_df = None
        self.submission_df = None
        
        self.user_map = {}
        self.item_map = {}
        self.n_users_global = 0
        self.n_items_global = 0
        
        self.user_all_interactions_map = {}

        self._load_and_preprocess_data()
        self._build_user_all_interactions_map()

    def _load_and_preprocess_data(self):
        """Loads and preprocesses raw CSV files."""
        print("Loading data...")
        ratings_path = os.path.join(self.data_dir, "train_ratings.csv")
        wishlist_path = os.path.join(self.data_dir, "train_tbr.csv")
        submission_path = os.path.join(self.data_dir, "sample_submission.csv")
        
        try:
            raw_ratings_df = pd.read_csv(ratings_path)
            if "rating" not in raw_ratings_df.columns:
                raise ValueError("Critical: 'rating' column is missing from train_ratings.csv.")

            raw_ratings_df[["sid_str", "pid_str"]] = raw_ratings_df["sid_pid"].str.split("_", expand=True)
            raw_ratings_df = raw_ratings_df.drop("sid_pid", axis=1)
            raw_ratings_df["sid"] = raw_ratings_df["sid_str"].astype(int)
            raw_ratings_df["pid"] = raw_ratings_df["pid_str"].astype(int)
            raw_ratings_df["rating"] = raw_ratings_df["rating"].astype(float)

            unique_sids_ratings = sorted(raw_ratings_df['sid'].unique())
            unique_pids_ratings = sorted(raw_ratings_df['pid'].unique())
            
            self.user_map = {sid: i for i, sid in enumerate(unique_sids_ratings)}
            self.item_map = {pid: i for i, pid in enumerate(unique_pids_ratings)}
            
            self.n_users_global = len(unique_sids_ratings)
            self.n_items_global = len(unique_pids_ratings)
            
            raw_ratings_df['user_id'] = raw_ratings_df['sid'].map(self.user_map)
            raw_ratings_df['item_id'] = raw_ratings_df['pid'].map(self.item_map)
            
            self.ratings_df = raw_ratings_df[['user_id', 'item_id', 'rating', 'sid', 'pid']].copy()
            self.ratings_df.dropna(subset=['user_id', 'item_id'], inplace=True)
            self.ratings_df['user_id'] = self.ratings_df['user_id'].astype(int)
            self.ratings_df['item_id'] = self.ratings_df['item_id'].astype(int)

        except Exception as e:
            print(f"Error loading ratings: {e}")
            return
        
        try:
            raw_wishlist_df = pd.read_csv(wishlist_path)
            raw_wishlist_df["sid"] = raw_wishlist_df["sid"].astype(int)
            raw_wishlist_df["pid"] = raw_wishlist_df["pid"].astype(int)

            raw_wishlist_df['user_id'] = raw_wishlist_df['sid'].map(self.user_map)
            raw_wishlist_df['item_id'] = raw_wishlist_df['pid'].map(self.item_map)
            
            raw_wishlist_df.dropna(subset=['user_id', 'item_id'], inplace=True)
            raw_wishlist_df['user_id'] = raw_wishlist_df['user_id'].astype(int)
            raw_wishlist_df['item_id'] = raw_wishlist_df['item_id'].astype(int)
            
            self.wishlist_df = raw_wishlist_df[['user_id', 'item_id', 'sid', 'pid']].copy()
            
        except Exception as e:
            print(f"Error loading wishlist: {e}")
            return

        try:
            raw_submission_df = pd.read_csv(submission_path)
            raw_submission_df[["sid_str", "pid_str"]] = raw_submission_df["sid_pid"].str.split("_", expand=True)
            raw_submission_df["sid"] = raw_submission_df["sid_str"].astype(int)
            raw_submission_df["pid"] = raw_submission_df["pid_str"].astype(int)
            
            raw_submission_df['user_id'] = raw_submission_df['sid'].map(self.user_map)
            raw_submission_df['item_id'] = raw_submission_df['pid'].map(self.item_map)
            
            self.submission_df = raw_submission_df[['sid_pid', 'user_id', 'item_id', 'sid', 'pid']].copy()

        except Exception as e:
            print(f"Error loading sample submission: {e}")
            return
        if self.ratings_df is not None and not self.ratings_df.empty:
            print(f"Loaded {len(self.ratings_df)} ratings.")
        if self.wishlist_df is not None and not self.wishlist_df.empty:
            print(f"Loaded {len(self.wishlist_df)} wishlist entries.")
        if self.submission_df is not None and not self.submission_df.empty:
            print(f"Loaded {len(self.submission_df)} submission pairs to predict.")
            
        print(f"Number of unique users (from ratings for mapping): {self.n_users_global}")
        print(f"Number of unique items (from ratings for mapping): {self.n_items_global}")

    def _build_user_all_interactions_map(self):
        """
        Builds a map of users to all items they have interacted with (rated or wishlisted).
        Keys are internal user_ids, values are sets of internal item_ids.
        """
        print("Building user all interactions map...")
        self.user_all_interactions_map = {u: set() for u in range(self.n_users_global)}

        if self.ratings_df is not None and not self.ratings_df.empty:
            for _, row in self.ratings_df.iterrows():
                if pd.notna(row['user_id']) and pd.notna(row['item_id']):
                    self.user_all_interactions_map[int(row['user_id'])].add(int(row['item_id']))
        
        if self.wishlist_df is not None and not self.wishlist_df.empty:
            for _, row in self.wishlist_df.iterrows():
                 if pd.notna(row['user_id']) and pd.notna(row['item_id']):
                    if int(row['user_id']) not in self.user_all_interactions_map:
                         pass
                    self.user_all_interactions_map[int(row['user_id'])].add(int(row['item_id']))
        print("User all interactions map built.")

    def get_ratings_df(self) -> pd.DataFrame:
        return self.ratings_df

    def get_wishlist_df(self) -> pd.DataFrame:
        if self.wishlist_df is not None and not self.wishlist_df.empty:
            return self.wishlist_df[self.wishlist_df['user_id'].notna()].copy()
        return pd.DataFrame(columns=['user_id', 'item_id', 'sid', 'pid'])


    def get_submission_df(self) -> pd.DataFrame:
        return self.submission_df

    def get_user_map(self):
        return self.user_map
        
    def get_item_map(self):
        return self.item_map

    def get_num_users(self) -> int:
        return self.n_users_global

    def get_num_items(self) -> int:
        return self.n_items_global
        
    def get_user_all_interactions(self) -> dict:
        return self.user_all_interactions_map

class RatingsDataset(Dataset):
    def __init__(self, users_tensor: torch.Tensor, items_tensor: torch.Tensor, ratings_tensor: torch.Tensor = None):
        self.users = users_tensor
        self.items = items_tensor
        self.ratings = ratings_tensor 

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if self.ratings is not None:
            return self.users[idx], self.items[idx], self.ratings[idx]
        else:
            return self.users[idx], self.items[idx]

class WishlistBPRDataset(Dataset):
    def __init__(self, 
                 wishlist_df: pd.DataFrame, 
                 user_all_interactions: dict, 
                 n_items_global: int, 
                 num_negative_samples: int = 1):
        """
        Dataset for BPR loss based on wishlist.
        Args:
            wishlist_df (pd.DataFrame): DataFrame with 'user_id' and 'item_id' (positive item).
                                        Assumes user_id and item_id are already mapped to global indices.
            user_all_interactions (dict): Map from user_id to a set of all item_ids they interacted with.
            n_items_global (int): Total number of unique items in the dataset.
            num_negative_samples (int): Number of negative items to sample for each positive instance.
        """
        if wishlist_df.empty or 'user_id' not in wishlist_df.columns or 'item_id' not in wishlist_df.columns:
            print("Warning: WishlistBPRDataset initialized with empty or invalid wishlist_df.")
            self.users = torch.empty(0, dtype=torch.long)
            self.positive_items = torch.empty(0, dtype=torch.long)
        else:
            valid_wishlist_df = wishlist_df.dropna(subset=['user_id', 'item_id'])
            self.users = torch.tensor(valid_wishlist_df['user_id'].values, dtype=torch.long)
            self.positive_items = torch.tensor(valid_wishlist_df['item_id'].values, dtype=torch.long)
            
        self.user_all_interactions = user_all_interactions
        self.n_items_global = n_items_global
        self.num_negative_samples = num_negative_samples


    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx].item()
        
        interacted_items = self.user_all_interactions.get(user_id, set())
        
        negative_samples = []
        if self.n_items_global > 0:
            attempts = 0
            max_attempts_per_sample = self.n_items_global * 2 # heuristic

            for _ in range(self.num_negative_samples):
                current_sample_attempts = 0
                while attempts < max_attempts_per_sample * self.num_negative_samples:
                    negative_item_candidate = random.randint(0, self.n_items_global - 1)
                    if negative_item_candidate not in interacted_items:
                        negative_samples.append(negative_item_candidate)
                        break
                    current_sample_attempts += 1
                    attempts +=1
                if current_sample_attempts >= max_attempts_per_sample and len(negative_samples) < self.num_negative_samples:
                    # fallback: if truly hard to find a negative sample, pick a random one not positive. very rare
                    pass


            while len(negative_samples) < self.num_negative_samples and self.n_items_global > 0:
                fallback_sample = random.randint(0, self.n_items_global - 1)
                negative_samples.append(fallback_sample)


        if not negative_samples:
            negative_samples = [0] * self.num_negative_samples

        return self.users[idx], self.positive_items[idx], torch.tensor(negative_samples, dtype=torch.long)