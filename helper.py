from typing import Tuple, Callable

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import Dataset, DataLoader
import os

#Seed need to be set for all experiments
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_DIR = "/Users/louis/Projects/BayesWatch_CollaborativeFilterning/Data"


def read_data_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads in data and splits it into training and validation sets with a 75/25 split."""
    
    df = pd.read_csv(os.path.join(DATA_DIR, "train_ratings.csv"))

    # Split sid_pid into sid and pid columns
    df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
    df = df.drop("sid_pid", axis=1)
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    
    # Split into train and validation dataset
    train_df, valid_df = train_test_split(df, test_size=0.25)
    return train_df, valid_df


def read_data_matrix(df: pd.DataFrame) -> np.ndarray:
    """Returns matrix view of the training data, where columns are scientists (sid) and
    rows are papers (pid)."""

    return df.pivot(index="sid", columns="pid", values="rating").values


def evaluate(valid_df: pd.DataFrame, pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Inputs:
        valid_df: Validation data, returned from read_data_df for example.
        pred_fn: Function that takes in arrays of sid and pid and outputs their rating predictions.

    Outputs: Validation RMSE
    """
    
    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)


def make_submission(pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], filename: os.PathLike):
    """Makes a submission CSV file that can be submitted to kaggle.

    Inputs:
        pred_fn: Function that takes in arrays of sid and pid and outputs a score.
        filename: File to save the submission to.
    """
    
    df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # Get sids and pids
    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0]
    pids = sid_pid[1]
    sids = sids.astype(int).values
    pids = pids.astype(int).values
    
    df["rating"] = pred_fn(sids, pids)
    df.to_csv(filename, index=False)

########### Helper for DMF



class RowDataset(Dataset):
    """
    Dataset yielding each row (scientist) vector from the rating matrix.
    """
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    def __len__(self):
        return self.matrix.size(0)

    def __getitem__(self, idx):
        return self.matrix[idx]


class ColumnDataset(Dataset):
    """
    Dataset yielding each column (paper) vector from the rating matrix.
    """
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix

    def __len__(self):
        return self.matrix.size(1)

    def __getitem__(self, idx):
        # returns the j-th column as a 1D tensor
        return self.matrix[:, idx]


def get_dataloaders(
    matrix: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns two DataLoaders:
      - row_loader: yields batches of row vectors x_i in R^P
      - col_loader: yields batches of column vectors y_j in R^S

    Args:
        matrix: torch.Tensor of shape (S, P)
        batch_size: batch size for both loaders
        shuffle: whether to shuffle samples
    """
    num_rows, num_cols = matrix.size(0), matrix.size(1)

    row_ds = RowDataset(matrix)
    col_ds = ColumnDataset(matrix)
    row_loader = DataLoader(row_ds, batch_size=num_rows, shuffle=shuffle)
    col_loader = DataLoader(col_ds, batch_size=num_cols, shuffle=shuffle)
    return row_loader, col_loader
