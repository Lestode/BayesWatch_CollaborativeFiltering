import os
import argparse
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch_geometric.data import HeteroData


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loading and Graph Construction
def read_data(data_dir: str):
    ratings = pd.read_csv(os.path.join(data_dir, 'train_ratings.csv'))
    ratings[['sid', 'pid']] = ratings['sid_pid'].str.split('_', expand=True).astype(int)
    ratings = ratings[['sid', 'pid', 'rating']]

    wishlist = pd.read_csv(os.path.join(data_dir, 'train_tbr.csv'))
    wishlist[['sid', 'pid']] = wishlist[['sid', 'pid']].astype(int)

    sample = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    sample[['sid', 'pid']] = sample['sid_pid'].str.split('_', expand=True).astype(int)

    return ratings, wishlist, sample


def build_hetero_graph(ratings: pd.DataFrame, wishlist: pd.DataFrame,
                       num_users: int, num_papers: int) -> HeteroData:
    data = HeteroData()
    data['user'].num_nodes = num_users
    data['paper'].num_nodes = num_papers

    src_r = torch.tensor(ratings['sid'].values, dtype=torch.long)
    dst_r = torch.tensor(ratings['pid'].values, dtype=torch.long)
    data['user', 'rates', 'paper'].edge_index = torch.stack([src_r, dst_r], dim=0)
    data['user', 'rates', 'paper'].edge_attr = torch.tensor(ratings['rating'].values, dtype=torch.float)

    src_w = torch.tensor(wishlist['sid'].values, dtype=torch.long)
    dst_w = torch.tensor(wishlist['pid'].values, dtype=torch.long)
    data['user', 'wants', 'paper'].edge_index = torch.stack([src_w, dst_w], dim=0)

    data['paper', 'rev_rates', 'user'].edge_index = data['user', 'rates', 'paper'].edge_index[[1, 0]].clone()
    data['paper', 'rev_wants', 'user'].edge_index = data['user', 'wants', 'paper'].edge_index[[1, 0]].clone()

    return data

# Model Definition (LightGCN-inspired)
class LightHeteroCF(nn.Module):
    def __init__(self, num_users: int, num_papers: int,
                 emb_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.paper_emb = nn.Embedding(num_papers, emb_dim)
        self.num_layers = num_layers

    def forward(self, data: HeteroData):
        x_u = self.user_emb.weight
        x_p = self.paper_emb.weight
        all_u, all_p = [x_u], [x_p]

        er = data['user', 'rates', 'paper'].edge_index
        ew = data['user', 'wants', 'paper'].edge_index
        rr = data['paper', 'rev_rates', 'user'].edge_index
        rw = data['paper', 'rev_wants', 'user'].edge_index

        for _ in range(self.num_layers):
            ru = self.propagate(x_u, x_p, er)
            wu = self.propagate(x_u, x_p, ew)
            x_u = 0.5 * (ru + wu)

            rp = self.propagate(x_p, x_u, rr)
            wp = self.propagate(x_p, x_u, rw)
            x_p = 0.5 * (rp + wp)

            all_u.append(x_u)
            all_p.append(x_p)

        u_final = torch.stack(all_u, dim=0).mean(dim=0)
        p_final = torch.stack(all_p, dim=0).mean(dim=0)
        return u_final, p_final

    @staticmethod
    def propagate(x_src: torch.Tensor, x_dst: torch.Tensor,
                  edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index
        messages = x_dst[dst_idx]
        deg = torch.bincount(src_idx, minlength=x_src.size(0)).clamp(min=1).unsqueeze(-1).float()
        agg = torch.zeros_like(x_src)
        agg.scatter_add_(0, src_idx.unsqueeze(-1).expand_as(messages), messages)
        return agg / deg

# Training and Evaluation
def train_one(model: nn.Module, data: HeteroData,
              lr: float, epochs: int, l1_reg: float) -> nn.Module:
    model.to(DEVICE)
    data = data.to(DEVICE)

    rates_idx = data['user', 'rates', 'paper'].edge_index
    true_r = data['user', 'rates', 'paper'].edge_attr
    wants_idx = data['user', 'wants', 'paper'].edge_index
    u_w = wants_idx[0]

    num_users = model.user_emb.num_embeddings
    num_papers = model.paper_emb.num_embeddings

    # Build exclusion sets
    excluded = [set() for _ in range(num_users)]
    r_cpu = rates_idx.cpu().numpy()
    tr_cpu = true_r.cpu().numpy()
    w_cpu = wants_idx.cpu().numpy()
    for u, p, r in zip(r_cpu[0], r_cpu[1], tr_cpu):
        if r > 3.0:
            excluded[u].add(p)
    for u, p in zip(w_cpu[0], w_cpu[1]):
        excluded[u].add(p)

    # Precompute neg candidates list and lengths
    all_papers = set(range(num_papers))
    neg_lists = [list(all_papers - s) for s in excluded]
    neg_len_cpu = [len(lst) for lst in neg_lists]
    max_len = max(neg_len_cpu)

    # Build negation matrix
    neg_mat = torch.zeros((num_users, max_len), dtype=torch.long, device=DEVICE)
    neg_len = torch.tensor(neg_len_cpu, dtype=torch.long, device=DEVICE)
    for u, lst in enumerate(neg_lists):
        if lst:
            neg_mat[u, :len(lst)] = torch.tensor(lst, dtype=torch.long, device=DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        u_emb, p_emb = model(data)

        # Rating loss with logistic link
        raw_r = (u_emb[rates_idx[0]] * p_emb[rates_idx[1]]).sum(dim=1)
        pred_r = 1.0 + 4.0 * torch.sigmoid(raw_r)
        loss_r = mse_loss(pred_r, true_r)

        # BPR loss with vectorized neg sampling
        rand = torch.rand(u_w.size(0), device=DEVICE)
        idx = (rand * neg_len[u_w]).long()
        neg_p = neg_mat[u_w, idx]
        pos_raw = (u_emb[u_w] * p_emb[wants_idx[1]]).sum(dim=1)
        neg_raw = (u_emb[u_w] * p_emb[neg_p]).sum(dim=1)
        loss_w = -torch.log(torch.sigmoid(pos_raw - neg_raw) + 1e-8).mean()

        # L1 regularization on embeddings
        l1_penalty = l1_reg * (model.user_emb.weight.abs().sum() + model.paper_emb.weight.abs().sum())

        (loss_r + loss_w + l1_penalty).backward()
        optimizer.step()

    return model


def compute_rmse(u_emb: torch.Tensor, p_emb: torch.Tensor,
                 val_df: pd.DataFrame) -> float:
    with torch.no_grad():
        sids = torch.tensor(val_df['sid'].values, dtype=torch.long, device=DEVICE)
        pids = torch.tensor(val_df['pid'].values, dtype=torch.long, device=DEVICE)
        raw = (u_emb[sids] * p_emb[pids]).sum(dim=1)
        preds = (1.0 + 4.0 * torch.sigmoid(raw)).cpu().numpy()
    return float(np.sqrt(mean_squared_error(val_df['rating'].values, preds)))

# Submission
def make_submission(model: nn.Module, sample_df: pd.DataFrame,
                    data: HeteroData, output_path: str) -> None:
    model.to(DEVICE)
    data = data.to(DEVICE)
    model.eval()
    with torch.no_grad():
        u_emb, p_emb = model(data)
        sids = torch.tensor(sample_df['sid'].values, dtype=torch.long, device=DEVICE)
        pids = torch.tensor(sample_df['pid'].values, dtype=torch.long, device=DEVICE)
        raw = (u_emb[sids] * p_emb[pids]).sum(dim=1)
        preds = (1.0 + 4.0 * torch.sigmoid(raw)).cpu().numpy()

    out = sample_df.copy()
    out['rating'] = preds
    out[['sid_pid', 'rating']].to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")

# Cross-Validation in the main code
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--emb_dims', type=str, default='64,128,256')
    parser.add_argument('--num_layers', type=str, default='3,4,5,6')
    parser.add_argument('--lrs', type=str, default='1e-3,5e-4,1e-4')
    parser.add_argument('--l1_reg', type=float, default=0.0, help='L1 regularization weight')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--val_split', type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ratings, wishlist, sample_df = read_data(args.data_dir)
    num_users = int(max(ratings['sid'].max(), wishlist['sid'].max()) + 1)
    num_papers = int(max(ratings['pid'].max(), wishlist['pid'].max()) + 1)

    emb_list = [int(x) for x in args.emb_dims.split(',')]
    layers_list = [int(x) for x in args.num_layers.split(',')]
    lr_list = [float(x) for x in args.lrs.split(',')]
    grid = list(itertools.product(emb_list, layers_list, lr_list))

    print(f'Device: {DEVICE}')
    results = []

    for emb_dim, nlayers, lr in grid:
        train_df, val_df = train_test_split(ratings, test_size=args.val_split,
                                            random_state=SEED)
        data_train = build_hetero_graph(train_df, wishlist, num_users, num_papers)
        model = LightHeteroCF(num_users, num_papers, emb_dim, nlayers)
        model = train_one(model, data_train, lr, args.epochs, args.l1_reg)

        data_eval = build_hetero_graph(train_df, wishlist, num_users, num_papers)
        u_emb, p_emb = model(data_eval.to(DEVICE))
        rmse = compute_rmse(u_emb, p_emb, val_df)

        fname = f"model_e{emb_dim}_l{nlayers}_lr{lr:.0e}_rmse{rmse:.4f}.pt"
        path = os.path.join(args.output_dir, fname)
        torch.save(model.state_dict(), path)
        results.append({'emb_dim': emb_dim, 'num_layers': nlayers,
                        'lr': lr, 'rmse': rmse, 'model_file': fname})

    df_res = pd.DataFrame(results)
    cv_path = os.path.join(args.output_dir, 'cv_results.csv')
    df_res.to_csv(cv_path, index=False)
    print(f"CV results saved to {cv_path}")

    best = df_res.loc[df_res['rmse'].idxmin()]
    print("Best hyperparams:", best.to_dict())

    data_full = build_hetero_graph(ratings, wishlist, num_users, num_papers)
    best_model = LightHeteroCF(num_users, num_papers,
                               int(best['emb_dim']), int(best['num_layers']))
    best_model = train_one(best_model, data_full, best['lr'], args.epochs, args.l1_reg)
    submission_path = os.path.join(args.output_dir, 'submission_best.csv')
    make_submission(best_model, sample_df, data_full, submission_path)

if __name__ == '__main__':
    main()
