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

# Utilities: Data Loading and Graph Preparation

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

# Dataset Class
class RatingEdgeDataset(Dataset):
    def __init__(self, train_pos, ratings):
        self.u, self.v = train_pos
        self.y = ratings
        assert len(self.u) == len(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.u[idx], self.v[idx], self.y[idx]

# Model Definition: LightGCN
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

# Training and Inference
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

    return np.sqrt(total_loss / n_samples) if n_samples > 0 else 0.0

def inference(model, data, sample_df, user_map, item_map, device, batch_size):
    model.eval()
    h = model.get_embeddings(data.adj.to(device))
    # map all sid_pid → indices
    sample_df = sample_df.copy()
    sample_df['rating'] = 3.0
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

# Main Training Loop
def main():
    wandb.init()
    cfg = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load everything
    ratings, wishlist, sample = load_csvs(cfg.data_dir)
    pu, qi, user_map, item_map = load_svd_embeddings(cfg.svd_path)
    data = build_graph(pu, qi, user_map, item_map, ratings, wishlist)

    # DataLoader
    dataset = RatingEdgeDataset(data.train_pos, data.y)
    loader = DataLoader(dataset,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        num_workers=cfg.num_workers,
                        pin_memory=True)

    # Model + optimizer
    initial_embeds = data.x
    model = LightGCN(cfg.num_layers,
                     cfg.dropout,
                     initial_embeds,
                     fine_tune_embed=cfg.fine_tune_embed).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    # Training
    for epoch in range(1, cfg.epochs+1):
        rmse = train_one_epoch(model, data, loader, optimizer, loss_fn, device, cfg.fine_tune_embed)
        wandb.log({'train_rmse': rmse, 'epoch': epoch})
        print(f"Epoch {epoch}/{cfg.epochs} RMSE: {rmse:.4f}")



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
    # build filename
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
    print(f"Saved final checkpoint: {ckpt_path}")

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
