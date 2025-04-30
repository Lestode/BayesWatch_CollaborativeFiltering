import os
import pandas as pd
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ---------------------------------------------
# Utilities: Data Loading and Graph Preparation
# ---------------------------------------------

def load_csvs(data_dir):
    ratings = pd.read_csv(
        os.path.join(data_dir, 'train_ratings.csv'),
        usecols=['sid_pid','rating'], dtype={'sid_pid':str,'rating':float}
    )
    ratings[['sid','pid']] = ratings['sid_pid'].str.split('_', expand=True).astype(int)

    wishlist = pd.read_csv(
        os.path.join(data_dir, 'train_tbr.csv'),
        usecols=['sid','pid'], dtype=int
    )

    sample = pd.read_csv(
        os.path.join(data_dir, 'sample_submission.csv'),
        dtype={'sid_pid':str}
    )
    sample[['sid','pid']] = sample['sid_pid'].str.split('_', expand=True).astype(int)

    return ratings, wishlist, sample

def load_svd_embeddings(svd_path):
    """
    Load precomputed SVD++ embeddings and mapping dicts.
    Returns pu, qi, user_map, item_map.
    """
    try:
        # PyTorch ≥2.0
        data = torch.load(svd_path, weights_only=False, map_location='cpu')
    except TypeError:
        # Older PyTorch
        data = torch.load(svd_path, map_location='cpu')

    for key in ('pu','qi','user_map','item_map'):
        if key not in data:
            raise KeyError(f"Missing key '{key}' in SVD embeddings file")

    pu = (torch.from_numpy(data['pu']).float()
          if isinstance(data['pu'], np.ndarray)
          else data['pu'].float())
    qi = (torch.from_numpy(data['qi']).float()
          if isinstance(data['qi'], np.ndarray)
          else data['qi'].float())

    return pu, qi, data['user_map'], data['item_map']


def build_graph(pu, qi, user_map, item_map, ratings, wishlist):
    n_users, n_items = pu.size(0), qi.size(0)
    x = torch.cat([pu, qi], dim=0)

    # rating edges
    u_r, v_r, rating_vals = [], [], []
    for sid, pid, r in ratings[['sid','pid','rating']].itertuples(index=False):
        if sid in user_map and pid in item_map:
            u = user_map[sid]
            v = item_map[pid] + n_users
            u_r.append(u); v_r.append(v); rating_vals.append(r)
    if not u_r:
        raise ValueError("No valid rating edges found")
    edge_r = torch.tensor([u_r + v_r, v_r + u_r], dtype=torch.long)
    y = torch.tensor(rating_vals, dtype=torch.float)

    # wishlist edges
    u_w, v_w = [], []
    for sid, pid in wishlist.itertuples(index=False):
        if sid in user_map and pid in item_map:
            u_w.append(user_map[sid])
            v_w.append(item_map[pid] + n_users)

    edges = edge_r
    if u_w:
        edge_w = torch.tensor([u_w + v_w, v_w + u_w], dtype=torch.long)
        edges = torch.cat([edge_r, edge_w], dim=1)

    edge_index = to_undirected(edges)
    num_nodes = n_users + n_items
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)
    deg_inv_sqrt = (deg + 1e-9).pow(-0.5)
    weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(edge_index, weight, (num_nodes, num_nodes)).coalesce()

    data = Data(x=x)
    data.adj = adj
    data.train_pos = torch.tensor([u_r, v_r], dtype=torch.long)
    data.y = y
    data.n_users, data.n_items = n_users, n_items
    return data

# ---------------------------------------------
# Dataset Class
# ---------------------------------------------
class RatingEdgeDataset(Dataset):
    def __init__(self, train_pos, ratings):
        self.u, self.v = train_pos
        self.y = ratings
        assert len(self.u) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.u[idx], self.v[idx], self.y[idx]

# ---------------------------------------------
# Model Definition: LightGCN
# ---------------------------------------------
class LightGCN(nn.Module):
    def __init__(self, n_layers, dropout, initial_embeds, fine_tune_embed=False):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        if fine_tune_embed:
            self.embeds = nn.Parameter(initial_embeds)
        else:
            self.register_buffer('embeds', initial_embeds)

    def forward(self, adj, u_idx, v_idx):
        h = self.propagate(adj)
        scores = (h[u_idx] * h[v_idx]).sum(dim=1)
        return 1.0 + 4.0 * torch.sigmoid(scores)

    def propagate(self, adj):
        h = self.embeds
        embs = [h]
        for _ in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
            embs.append(h)
        h = torch.stack(embs, dim=0).mean(dim=0)
        # only apply dropout during training
        if self.training:
            h = self.dropout(h)
        return h

    def get_embeddings(self, adj):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            h = self.propagate(adj)
        if was_training:
            self.train()
        return h

# ---------------------------------------------
# Training and Inference
# ---------------------------------------------
def train_one_epoch(model, data, loader, optimizer, loss_fn, device, fine_tune_embed=False):
    model.train()
    total_loss = 0.0
    n_samples = len(loader.dataset)

    # precompute once if not fine-tuning
    if not fine_tune_embed:
        with torch.no_grad():
            h = model.get_embeddings(data.adj.to(device))

    for u, v, r in loader:
        u, v, r = u.to(device), v.to(device), r.to(device)
        optimizer.zero_grad()
        if fine_tune_embed:
            preds = model(data.adj.to(device), u, v)
        else:
            scores = (h[u] * h[v]).sum(dim=1)
            preds = 1.0 + 4.0 * torch.sigmoid(scores)
        loss = loss_fn(preds, r)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * u.size(0)

    # fixed: return RMSE, not call a float
    return np.sqrt(total_loss / n_samples) if n_samples > 0 else 0.0


def evaluate(model, adj, loader, device, fine_tune_embed):
    model.eval()
    mse_loss = nn.MSELoss(reduction='sum')
    total_loss = 0.0
    n    = len(loader.dataset)
    # precompute embeddings if not fine-tuning
    if not fine_tune_embed:
        with torch.no_grad():
            h = model.get_embeddings(adj)
    with torch.no_grad():
        for u, v, r in loader:
            u, v, r = u.to(device), v.to(device), r.to(device)
            if fine_tune_embed:
                preds = model(adj, u, v)
            else:
                scores = (h[u] * h[v]).sum(dim=1)
                preds = 1.0 + 4.0 * torch.sigmoid(scores)
            total_loss += mse_loss(preds, r).item()
    return np.sqrt(total_loss / n)

    
def inference(model, data, sample_df, user_map, item_map, device, batch_size):
    model.eval()
    h = model.get_embeddings(data.adj.to(device))
    # map all sid_pid → indices
    sample_df = sample_df.copy()
    sample_df['rating'] = 3.0  # default
    valid = []
    for i, (sid, pid) in sample_df[['sid','pid']].iterrows():
        if sid in user_map and pid in item_map:
            valid.append(i)
    if valid:
        u_idx = torch.tensor([user_map[sample_df.at[i,'sid']] for i in valid], device=device)
        v_idx = torch.tensor([item_map[sample_df.at[i,'pid']]+data.n_users for i in valid], device=device)
        preds = []
        for i in range(0, len(u_idx), batch_size):
            batch_u = u_idx[i:i+batch_size]
            batch_v = v_idx[i:i+batch_size]
            with torch.no_grad():
                preds.append(model(data.adj.to(device), batch_u, batch_v).cpu())
        preds = torch.cat(preds).numpy()
        sample_df.loc[valid, 'rating'] = preds
    return sample_df[['sid_pid','rating']]

def generate_submission(model, data, sample_path, output_path, user_map, item_map, device, batch_size):
    """
    Fill in sample_submission, defaulting unseen pairs to 3.0, and save.
    """
    sample = pd.read_csv(sample_path, dtype={'sid_pid':str})
    sample[['sid','pid']] = sample['sid_pid'].str.split('_', expand=True).astype(int)

    submission = inference(model, data, sample, user_map, item_map, device, batch_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

# ---------------------------------------------
# Main Training Loop
# ---------------------------------------------
def main():
    wandb.init()
    cfg = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. load full ratings, then split
    ratings, wishlist, sample = load_csvs(cfg.data_dir)
    train_r, test_r = train_test_split(
        ratings, test_size=0.1, random_state=42
    )

    # 2. build graph using ONLY train_r
    pu, qi, user_map, item_map = load_svd_embeddings(cfg.svd_path)
    data = build_graph(pu, qi, user_map, item_map, train_r, wishlist)
    adj = data.adj.to(device)

    # 3. Create train / test datasets & loaders
    train_dataset = RatingEdgeDataset(data.train_pos, data.y)
    train_loader  = DataLoader(train_dataset,
                               batch_size=cfg.batch_size,
                               shuffle=True,
                               num_workers=cfg.num_workers,
                               pin_memory=True)
    # map test edges into tensors
    test_u, test_v, test_y = [], [], []
    for sid, pid, r in test_r[['sid','pid','rating']].itertuples(index=False):
        if sid in user_map and pid in item_map:
            test_u.append(user_map[sid])
            test_v.append(item_map[pid] + data.n_users)
            test_y.append(r)
    test_pos = (torch.tensor(test_u), torch.tensor(test_v))
    test_dataset = RatingEdgeDataset(test_pos, torch.tensor(test_y))
    test_loader  = DataLoader(test_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers,
                              pin_memory=True)

    # 4. Model + optimizer
    model = LightGCN(cfg.num_layers, cfg.dropout,
                     data.x.to(device),
                     fine_tune_embed=cfg.fine_tune_embed).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    # 5. Training loop with validation
    for epoch in range(1, cfg.epochs+1):
        train_rmse = train_one_epoch(
            model, data, train_loader, optimizer,
            nn.MSELoss(), device, cfg.fine_tune_embed
        )
        val_rmse = evaluate(
            model, adj, test_loader, device, cfg.fine_tune_embed
        )

        wandb.log({
            'train_rmse': train_rmse,
            'val_rmse':   val_rmse,
            'epoch':      epoch
        })
        print(f"Epoch {epoch}/{cfg.epochs} | "
              f"Train RMSE: {train_rmse:.4f} | "
              f"Val RMSE: {val_rmse:.4f}")



    # Save final model checkpoint with hyperparameters
    run_id = wandb.run.id if wandb.run is not None else 'local'
    hp = {
        'layers': cfg.num_layers,
        'dropout': cfg.dropout,
        'lr': cfg.lr,
        'wd': cfg.weight_decay,
        'finetune': int(cfg.fine_tune_embed),
        'bs': cfg.batch_size
    }
    # build filename like: lightgcn_r<run_id>_L3_D0.5_lr1e-3_wd1e-5_ft0_bs256.pt
    hp_str = (
        f"L{hp['layers']}_D{hp['dropout']}_lr{hp['lr']}_wd{hp['wd']}"
        f"_ft{hp['finetune']}_bs{hp['bs']}"
    )
    ckpt_name = f"lightgcn_r{run_id}_{hp_str}_final.pt"
    ckpt_path = os.path.join(cfg.output_dir, ckpt_name)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(cfg),
        'epoch': cfg.epochs
    }, ckpt_path)
    print(f"🔖 Saved final checkpoint: {ckpt_path}")

    # Inference + submission
    generate_submission(
        model, data,
        os.path.join(cfg.data_dir, 'sample_submission.csv'),
        os.path.join(cfg.output_dir, f'submission_{hp_str}.csv'),
        user_map, item_map,
        device, batch_size=cfg.batch_size * 4
    )

    wandb.finish()

if __name__ == '__main__':
    main()
