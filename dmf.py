import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
# Optional: remove or keep wandb based on your preference

from helper import read_data_df, read_data_matrix, evaluate  # Assumes these functions exist

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

        def build_mlp(dims):
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

    def forward(self, row, col):
        u = self.user_net(row)
        v = self.item_net(col)
        sim = F.cosine_similarity(u, v, dim=-1)
        return sim.clamp(min=1e-6)

class InteractionDataset(Dataset):
    """
    Yields (user_idx, item_idx, rating) with negative sampling.
    """
    def __init__(self, matrix: torch.Tensor, neg_ratio: int = 1):
        super().__init__()
        pos = (matrix > 0).nonzero(as_tuple=False)
        neg = (matrix == 0).nonzero(as_tuple=False)
        pos_idx = pos.tolist()
        num_neg = len(pos_idx) * neg_ratio
        neg_idx = neg[torch.randperm(len(neg))[:num_neg]].tolist()

        self.interactions = [(i, j, matrix[i, j].item()) for i, j in pos_idx]
        self.interactions += [(i, j, 0.0) for i, j in neg_idx]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]


def train_dmf(matrix: torch.Tensor, config, device):
    """
    Train the DMF model.
    """
    matrix = matrix.to(device)
    num_users, num_items = matrix.size()

    model = DMFModel(
        num_user_inputs = num_items,
        num_item_inputs = num_users,
        hidden_dim1      = config.hidden_dim1,
        hidden_dim2      = config.hidden_dim2,
        embedding_dim    = config.embedding_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataset = InteractionDataset(matrix, config.neg_ratio)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    max_rating = matrix.max().item()
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        for i_idxs, j_idxs, ratings in loader:
            i_idxs  = i_idxs.to(device, non_blocking=True)
            j_idxs  = j_idxs.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)

            # matrix already lives on `device`, so indexing is safe
            rows = matrix[i_idxs]
            cols = matrix[:, j_idxs].t()

            preds = model(rows, cols)
            normalized_labels = (ratings.float() / max_rating).clamp(min=1e-6)
            #norm_r = labels / max_rating
            #loss = F.binary_cross_entropy(preds, labels)
            loss = F.mse_loss(preds, normalized_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * ratings.size(0)

        epoch_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{config.epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({ 'train/epoch_loss': epoch_loss, 'epoch': epoch })

    return model


def main():
    parser = argparse.ArgumentParser(description="Train DMF model with manual config (no W&B sweeps)")
    parser.add_argument('--hidden_dim1', type=int, default=20)
    parser.add_argument('--hidden_dim2', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--neg_ratio', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model_path', type=str, default='dmf_weights.pth')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--wandb_project', type=str, default='cil_dmf')
    parser.add_argument('--wandb_entity', type=str, default='louis-barinka-eth-z-rich')
    args = parser.parse_args()
    print("ARGS DICT:", vars(args))
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    run.config.update(vars(args)) 

    # Load data
    train_df, valid_df = read_data_df()
    train_mat = np.nan_to_num(read_data_matrix(train_df), nan=0.0)
    train_tensor = torch.from_numpy(train_mat).float()

    # Train
    model = train_dmf(train_tensor, args, torch.device(args.device))

    # Save model weights
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

    # Define prediction function for evaluation
    def pred_fn_factory(train_tensor, model, max_rating, device):
        def pred_fn(sids, pids):
            model.eval()
            s = torch.from_numpy(sids).long()
            p = torch.from_numpy(pids).long()
            preds = []
            batch_size = 1024
            with torch.no_grad():
                for i in range(0, len(s), batch_size):
                    sb = s[i:i+batch_size]
                    pb = p[i:i+batch_size]
                    rows = train_tensor[sb].to(device)
                    cols = train_tensor[:, pb].t().to(device)
                    out = model(rows, cols).cpu().numpy()
                    preds.append(out)
            return np.concatenate(preds) * max_rating
        return pred_fn

    # Evaluate
    max_rating_val = train_mat.max().item()
    pred_fn = pred_fn_factory(train_tensor, model, max_rating_val, torch.device(args.device))
    rmse = evaluate(valid_df, pred_fn)
    print(f"Validation RMSE: {rmse:.4f}")
    wandb.log({'val/rmse': rmse})
    wandb.finish()

if __name__ == '__main__':
    main()