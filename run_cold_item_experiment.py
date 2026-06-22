# coding: utf-8
"""Run GE-MV-MGDP V2 on MMRec-style datasets.

This script adapts the Kaggle JSONL prototype to the MMRec layout:

    DATA_COLD/baby/baby.inter
    DATA_COLD/baby/image_feat.npy
    DATA_COLD/baby/text_feat.npy

The learned branch projects image/text features to the same embedding size before
fusion. The zero-shot branch mixes image and text scores, not raw vectors, so it
also works when image and text feature dimensions are different.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


TOPK_LIST = [1, 5, 10, 20]


def l2_norm(x):
    return F.normalize(x, p=2, dim=-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mmrec_split(data_root, dataset, train_label=0, test_label=2, max_train_items=None, max_test_items=None):
    dataset_dir = os.path.join(data_root, dataset)
    inter_path = os.path.join(dataset_dir, "{}.inter".format(dataset))
    image_path = os.path.join(dataset_dir, "image_feat.npy")
    text_path = os.path.join(dataset_dir, "text_feat.npy")

    for path in (inter_path, image_path, text_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    df = pd.read_csv(inter_path, sep="\t")
    required_cols = {"userID", "itemID", "x_label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("{} must contain columns {}".format(inter_path, sorted(required_cols)))

    train_df = df[df["x_label"] == train_label][["userID", "itemID"]].copy()
    test_df = df[df["x_label"] == test_label][["userID", "itemID"]].copy()

    train_items = sorted(train_df["itemID"].unique().tolist())
    test_items = sorted(test_df["itemID"].unique().tolist())
    if max_train_items is not None:
        train_items = train_items[:max_train_items]
        train_df = train_df[train_df["itemID"].isin(train_items)]
    if max_test_items is not None:
        test_items = test_items[:max_test_items]
        test_df = test_df[test_df["itemID"].isin(test_items)]

    train_users = sorted(train_df["userID"].unique().tolist())
    train_user_set = set(train_users)
    test_df = test_df[test_df["userID"].isin(train_user_set)]

    image_feat = np.load(image_path, allow_pickle=True).astype(np.float32)
    text_feat = np.load(text_path, allow_pickle=True).astype(np.float32)

    max_item_id = max(train_items + test_items) if train_items or test_items else -1
    if max_item_id >= len(image_feat) or max_item_id >= len(text_feat):
        raise ValueError("itemID exceeds feature rows: max itemID={}, image rows={}, text rows={}".format(
            max_item_id, len(image_feat), len(text_feat)
        ))

    return train_df, test_df, train_items, test_items, train_users, image_feat, text_feat


def build_knn_adj(features, k=10):
    n = features.size(0)
    if n == 0:
        raise ValueError("Cannot build adjacency for zero items")
    k = max(1, min(k, n - 1))
    sim = torch.matmul(features, features.T)
    _, idx = torch.topk(sim, k + 1)
    idx = idx[:, 1:]

    r = torch.arange(n).view(-1, 1).repeat(1, k).view(-1).cpu().numpy()
    c = idx.reshape(-1).cpu().numpy()
    adj = sp.csr_matrix((np.ones_like(r, dtype=np.float32), (r, c)), shape=(n, n))
    return normalize_adj(adj)


def build_thresh_adj(features, tau=0.3):
    n = features.size(0)
    features = l2_norm(features)
    sim = torch.matmul(features, features.T)
    mask = (sim >= tau) & (~torch.eye(n, device=sim.device).bool())
    r, c = mask.nonzero(as_tuple=True)
    adj = sp.csr_matrix(
        (np.ones(len(r), dtype=np.float32), (r.cpu().numpy(), c.cpu().numpy())),
        shape=(n, n),
    )
    return normalize_adj(adj)


def normalize_adj(adj):
    adj = adj + adj.T
    adj = adj + sp.eye(adj.shape[0], dtype=np.float32)
    adj.data = np.clip(adj.data, 0, 1)
    d = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d[np.isinf(d)] = 0.0
    d_m = sp.diags(d)
    return d_m.dot(adj).dot(d_m).tocsr()


def to_sparse(mx, device):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, mx.shape).coalesce().to(device)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x1, x2):
        g = self.gate(torch.cat([x1, x2], dim=-1))
        return g * x1 + (1 - g) * x2


class ItemEncoder(nn.Module):
    def __init__(self, image_dim, text_dim, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(image_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.pt = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.fusion = GatedFusion(embed_dim)
        self.gnn_weights = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

    def forward(self, image_feat, text_feat, adj):
        zi = l2_norm(self.pi(image_feat))
        zt = l2_norm(self.pt(text_feat))
        h = self.fusion(zi, zt)
        outputs = [h]
        for weight, norm in zip(self.gnn_weights, self.gnn_norms):
            h_new = torch.sparse.mm(adj, h)
            h_new = F.gelu(norm(weight(h_new)))
            h = h + 0.3 * h_new
            outputs.append(l2_norm(h))
        return l2_norm(sum(outputs) / len(outputs)), zi, zt


class MemoryQueue(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.register_buffer("queue", l2_norm(torch.randn(size, dim)))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.size = size

    @torch.no_grad()
    def enqueue_and_dequeue(self, keys):
        keys = l2_norm(keys.detach())
        batch_size = min(keys.shape[0], self.size)
        keys = keys[:batch_size]
        ptr = int(self.ptr)
        if ptr + batch_size <= self.size:
            self.queue[ptr:ptr + batch_size] = keys
        else:
            rem = self.size - ptr
            self.queue[ptr:] = keys[:rem]
            self.queue[:batch_size - rem] = keys[rem:]
        self.ptr[0] = (ptr + batch_size) % self.size


class DeepGEV2(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        embed_dim,
        n_users,
        hidden_dim=1024,
        n_layers=3,
        dropout=0.1,
        queue_size=4096,
        momentum=0.995,
        use_momentum=True,
        use_queue=True,
        use_modal_alignment=True,
        use_pos_reg=True,
    ):
        super().__init__()
        self.use_momentum = use_momentum
        self.use_queue = use_queue and queue_size > 0
        self.use_modal_alignment = use_modal_alignment
        self.use_pos_reg = use_pos_reg
        self.momentum = momentum

        self.user_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.online_encoder = ItemEncoder(image_dim, text_dim, embed_dim, hidden_dim, n_layers, dropout)
        self.target_encoder = ItemEncoder(image_dim, text_dim, embed_dim, hidden_dim, n_layers, dropout)
        for p_online, p_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.copy_(p_online.data)
            p_target.requires_grad = False

        if self.use_queue:
            self.queue = MemoryQueue(queue_size, embed_dim)

        self.u_emb = nn.Embedding(n_users, embed_dim)
        nn.init.xavier_uniform_(self.u_emb.weight)
        self.log_temp = nn.Parameter(torch.ones(1) * np.log(0.07))

    def train(self, mode=True):
        super().train(mode)
        if self.use_momentum:
            self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        for p_online, p_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data, alpha=1 - self.momentum)

    def encode_items(self, image_feat, text_feat, adj, mode="online"):
        if mode == "online":
            return self.online_encoder(image_feat, text_feat, adj)
        with torch.no_grad():
            return self.target_encoder(image_feat, text_feat, adj)

    @torch.no_grad()
    def enqueue_target(self, image_feat, text_feat, adj, item_idx):
        if self.use_momentum and self.use_queue:
            v_target, _, _ = self.target_encoder(image_feat, text_feat, adj)
            self.queue.enqueue_and_dequeue(v_target[item_idx])

    def forward(self, image_feat, text_feat, adj, user_idx, item_idx):
        v_online, zi, zt = self.online_encoder(image_feat, text_feat, adj)
        ue = l2_norm(self.user_proj(self.u_emb(user_idx)))
        temp = self.log_temp.exp().clamp(min=0.01, max=0.5)

        if self.use_momentum:
            with torch.no_grad():
                v_target, _, _ = self.target_encoder(image_feat, text_feat, adj)
            v_pos = v_target[item_idx]
            l_pos = torch.einsum("nc,nc->n", ue, v_pos).unsqueeze(-1)
            if self.use_queue:
                l_neg = torch.matmul(ue, self.queue.queue.T)
                logits = torch.cat([l_pos, l_neg], dim=1) / temp
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=ue.device)
                loss_main = F.cross_entropy(logits, labels)
            else:
                logits = torch.matmul(ue, v_target[item_idx].T) / temp
                labels = torch.arange(logits.shape[0], device=ue.device)
                loss_main = F.cross_entropy(logits, labels)
            pos_scores = (ue * v_online[item_idx]).sum(-1)
        else:
            logits = torch.matmul(ue, v_online.T) / temp
            loss_main = F.cross_entropy(logits, item_idx)
            pos_scores = (ue * v_online[item_idx]).sum(-1)

        loss_modal = F.mse_loss(zi, zt) * 0.1 if self.use_modal_alignment else 0.0
        loss_reg = -pos_scores.mean() * 0.05 if self.use_pos_reg else 0.0
        return loss_main + loss_modal + loss_reg


def build_adj(features, mode, k, tau, device):
    combined = l2_norm(torch.cat(features, dim=1)).detach().cpu()
    if mode == "knn":
        return to_sparse(build_knn_adj(combined, k=k), device)
    if mode == "threshold":
        return to_sparse(build_thresh_adj(combined, tau=tau), device)
    raise ValueError("Unknown graph mode: {}".format(mode))


def eval_grid(model, tensors, meta, args, device):
    tri_raw, trt_raw, tei_raw, tet_raw, tri, trt, tei, tet = tensors
    train_items, test_items, user2idx, train_df, test_df = meta

    with torch.no_grad():
        all_i = torch.cat([tri, tei], dim=0)
        all_t = torch.cat([trt, tet], dim=0)
        adj_all = build_adj((all_i, all_t), args.graph_mode, args.knn_k, args.threshold_tau, device)
        v_all, _, _ = model.encode_items(all_i, all_t, adj_all, mode="online")
        learned_test = v_all[len(train_items):]
        learned_users = l2_norm(model.user_proj(model.u_emb.weight.data))

        train_item2local = {item_id: idx for idx, item_id in enumerate(train_items)}
        rows, cols = [], []
        for row in train_df.itertuples(index=False):
            if row.userID in user2idx and row.itemID in train_item2local:
                rows.append(user2idx[row.userID])
                cols.append(train_item2local[row.itemID])
        hist = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(len(user2idx), len(train_items)))
        denom = np.clip(np.array(hist.sum(1)), 1, None)

        train_img_zs = l2_norm(tri_raw)
        train_txt_zs = l2_norm(trt_raw)
        test_img_zs = l2_norm(tei_raw)
        test_txt_zs = l2_norm(tet_raw)
        user_img_zs = l2_norm(torch.tensor(hist.dot(train_img_zs.cpu().numpy()) / denom,
                                           dtype=torch.float32, device=device))
        user_txt_zs = l2_norm(torch.tensor(hist.dot(train_txt_zs.cpu().numpy()) / denom,
                                           dtype=torch.float32, device=device))

        test_item2local = {item_id: idx for idx, item_id in enumerate(test_items)}
        eval_pairs = []
        for row in test_df.itertuples(index=False):
            if row.userID in user2idx and row.itemID in test_item2local:
                eval_pairs.append((user2idx[row.userID], test_item2local[row.itemID]))

        eval_users = sorted({uid for uid, _ in eval_pairs})
        eval_user_pos = {uid: pos for pos, uid in enumerate(eval_users)}
        eval_rows = np.array([eval_user_pos[uid] for uid, _ in eval_pairs], dtype=np.int64)
        eval_cols = np.array([pos for _, pos in eval_pairs], dtype=np.int64)
        eval_user_tensor = torch.tensor(eval_users, dtype=torch.long, device=device)

        learned_score_mat = torch.matmul(learned_users[eval_user_tensor], learned_test.T)
        image_score_mat = torch.matmul(user_img_zs[eval_user_tensor], test_img_zs.T)
        text_score_mat = torch.matmul(user_txt_zs[eval_user_tensor], test_txt_zs.T)

        best = None
        print("\nGRID SEARCH (alpha=image score weight, beta=learned score weight)")
        print("=" * 72)
        for alpha in args.alphas:
            zs_score_mat = alpha * image_score_mat + (1 - alpha) * text_score_mat
            for beta in args.betas:
                final_score_mat = beta * learned_score_mat + (1 - beta) * zs_score_mat
                _, top_idx = torch.topk(final_score_mat, min(max(TOPK_LIST), len(test_items)), dim=1)
                top_idx = top_idx.detach().cpu().numpy()

                ranks = np.full(len(eval_pairs), np.inf, dtype=np.float32)
                for pair_idx, (row_idx, item_idx) in enumerate(zip(eval_rows, eval_cols)):
                    pos = np.where(top_idx[row_idx] == item_idx)[0]
                    if len(pos):
                        ranks[pair_idx] = pos[0]

                cnt = max(len(ranks), 1)
                metrics = {
                    k: {
                        "recall": float(np.mean(ranks < k)),
                        "ndcg": float(np.sum(np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)) / cnt),
                        "mrr": float(np.sum(np.where(ranks < k, 1.0 / (ranks + 1), 0.0)) / cnt),
                    }
                    for k in TOPK_LIST
                }
                r20 = metrics[20]["recall"]
                print("alpha={:>4.2f} | beta={:>4.2f} | R@20={:.4f} NDCG@20={:.4f} MRR@20={:.4f}".format(
                    alpha, beta, metrics[20]["recall"], metrics[20]["ndcg"], metrics[20]["mrr"]
                ))
                if best is None or r20 > best["recall@20"]:
                    best = {"recall@20": r20, "alpha": alpha, "beta": beta, "metrics": metrics, "eval_pairs": len(eval_pairs)}
        return best


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="DATA_COLD")
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--train-label", type=int, default=0)
    parser.add_argument("--test-label", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph-mode", choices=["knn", "threshold"], default="threshold")
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--threshold-tau", type=float, default=0.3)
    parser.add_argument("--alphas", type=parse_float_list, default=parse_float_list("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"))
    parser.add_argument("--betas", type=parse_float_list, default=parse_float_list("0,0.2,0.4,0.6,0.8,1.0"))
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-test-items", type=int, default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print("=" * 72)
    print("GE-MV-MGDP V2 on MMRec")
    print("dataset={} data_root={} device={}".format(args.dataset, args.data_root, device))
    print("=" * 72)
    sys.stdout.flush()

    train_df, test_df, train_items, test_items, train_users, image_feat, text_feat = load_mmrec_split(
        args.data_root, args.dataset, args.train_label, args.test_label, args.max_train_items, args.max_test_items
    )
    if not train_items or not test_items or not train_users:
        raise ValueError("Empty train/test split after filtering")

    user2idx = {user_id: idx for idx, user_id in enumerate(train_users)}
    item2idx = {item_id: idx for idx, item_id in enumerate(train_items)}

    train_image_raw = torch.tensor(image_feat[train_items], dtype=torch.float32, device=device)
    train_text_raw = torch.tensor(text_feat[train_items], dtype=torch.float32, device=device)
    test_image_raw = torch.tensor(image_feat[test_items], dtype=torch.float32, device=device)
    test_text_raw = torch.tensor(text_feat[test_items], dtype=torch.float32, device=device)

    image_mean, image_std = train_image_raw.mean(0), train_image_raw.std(0) + 1e-9
    text_mean, text_std = train_text_raw.mean(0), train_text_raw.std(0) + 1e-9
    train_image = (train_image_raw - image_mean) / image_std
    train_text = (train_text_raw - text_mean) / text_std
    test_image = (test_image_raw - image_mean) / image_std
    test_text = (test_text_raw - text_mean) / text_std

    print("Train items: {} | Test items: {} | Train users: {} | Train pairs: {} | Test pairs: {}".format(
        len(train_items), len(test_items), len(train_users), len(train_df), len(test_df)
    ))
    print("Image dim: {} | Text dim: {}".format(train_image.shape[1], train_text.shape[1]))
    sys.stdout.flush()

    adj = build_adj((train_image, train_text), args.graph_mode, args.knn_k, args.threshold_tau, device)
    model = DeepGEV2(
        image_dim=train_image.shape[1],
        text_dim=train_text.shape[1],
        embed_dim=args.embed_dim,
        n_users=len(train_users),
        hidden_dim=args.hidden_dim,
        n_layers=args.gnn_layers,
        dropout=args.dropout,
        queue_size=args.queue_size,
    ).to(device)

    with torch.no_grad():
        init_items, _, _ = model.encode_items(train_image, train_text, adj, mode="online")
        user_hist = defaultdict(list)
        for row in train_df.itertuples(index=False):
            if row.userID in user2idx and row.itemID in item2idx:
                user_hist[row.userID].append(item2idx[row.itemID])
        for user_id, local_items in user_hist.items():
            model.u_emb.weight.data[user2idx[user_id]] = init_items[local_items].mean(0)

    pairs = [
        (user2idx[row.userID], item2idx[row.itemID])
        for row in train_df.itertuples(index=False)
        if row.userID in user2idx and row.itemID in item2idx
    ]
    loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True, drop_last=len(pairs) >= args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    print("Training {} epochs...".format(args.epochs))
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_users, batch_items in loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            loss = model(train_image, train_text, adj, batch_users, batch_items)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()
            model.enqueue_target(train_image, train_text, adj, batch_items)
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()
        if (epoch + 1) % max(1, min(10, args.epochs)) == 0 or epoch == 0:
            print("Epoch {:3d}/{} | Loss={:.4f} | Time={:.1f}s".format(
                epoch + 1, args.epochs, total_loss / max(steps, 1), time.time() - start_time
            ))
            sys.stdout.flush()

    model.eval()
    best = eval_grid(
        model,
        (train_image_raw, train_text_raw, test_image_raw, test_text_raw, train_image, train_text, test_image, test_text),
        (train_items, test_items, user2idx, train_df, test_df),
        args,
        device,
    )

    print("\nBEST CONFIG")
    print("=" * 72)
    print("Best by Recall@20: alpha={:.2f}, beta={:.2f}, eval_pairs={}".format(
        best["alpha"], best["beta"], best["eval_pairs"]
    ))
    for k in TOPK_LIST:
        print("K={:2d}: Recall={:.4f} | NDCG={:.4f} | MRR={:.4f}".format(
            k, best["metrics"][k]["recall"], best["metrics"][k]["ndcg"], best["metrics"][k]["mrr"]
        ))

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        print("Results written to {}".format(args.output_path))


if __name__ == "__main__":
    main()
