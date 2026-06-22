# coding: utf-8
"""Create random, user-time, and global-time MMRec split folders.

Labels follow the MMRec convention:
    x_label = 0 -> train
    x_label = 1 -> validation
    x_label = 2 -> test
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("baby", "sports", "elec", "clothing")
SIDE_FILES = (
    "image_feat.npy",
    "text_feat.npy",
    "u_id_mapping.csv",
    "i_id_mapping.csv",
    "user_graph_dict.npy",
    "mm_adj_freedomdsp_10_1.pt",
)


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_counts(n_rows, train_ratio=0.8, valid_ratio=0.1):
    if n_rows <= 0:
        return 0, 0, 0
    if n_rows == 1:
        return 1, 0, 0
    if n_rows == 2:
        return 1, 0, 1
    train_n = int(n_rows * train_ratio)
    valid_n = int(n_rows * valid_ratio)
    train_n = max(1, train_n)
    valid_n = max(1, valid_n)
    if train_n + valid_n >= n_rows:
        valid_n = 1
        train_n = n_rows - 2
    test_n = n_rows - train_n - valid_n
    return train_n, valid_n, test_n


def labels_for_count(n_rows):
    train_n, valid_n, test_n = split_counts(n_rows)
    return np.array([0] * train_n + [1] * valid_n + [2] * test_n, dtype=np.int64)


def random_per_user_split(df, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby("userID", sort=True):
        group = group.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).copy()
        group["x_label"] = labels_for_count(len(group))
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def user_time_split(df):
    parts = []
    for _, group in df.sort_values(["userID", "timestamp", "itemID"]).groupby("userID", sort=True):
        group = group.copy()
        group["x_label"] = labels_for_count(len(group))
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def global_time_split(df):
    df = df.sort_values(["timestamp", "userID", "itemID"]).copy()
    df["x_label"] = labels_for_count(len(df))
    train_users = set(df.loc[df["x_label"] == 0, "userID"].unique())
    df = df[(df["x_label"] == 0) | (df["userID"].isin(train_users))].copy()
    return df


def create_split(source_root, target_root, dataset, split_name, seed):
    source_dir = source_root / dataset
    target_dir = target_root / split_name / dataset
    inter_name = "{}.inter".format(dataset)
    source_inter = source_dir / inter_name
    target_inter = target_dir / inter_name
    if not source_inter.is_file():
        raise FileNotFoundError(source_inter)

    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_inter, sep="\t")
    required_cols = {"userID", "itemID", "timestamp"}
    if not required_cols.issubset(df.columns):
        raise ValueError("{} must contain {}".format(source_inter, sorted(required_cols)))
    if "x_label" in df.columns:
        df = df.drop(columns=["x_label"])

    if split_name == "random":
        out_df = random_per_user_split(df, seed)
    elif split_name == "user_time":
        out_df = user_time_split(df)
    elif split_name == "global_time":
        out_df = global_time_split(df)
    else:
        raise ValueError("Unknown split: {}".format(split_name))

    out_df.to_csv(target_inter, sep="\t", index=False)
    for name in SIDE_FILES:
        src = source_dir / name
        if src.is_file():
            link_or_copy(src, target_dir / name)

    counts = out_df["x_label"].value_counts().to_dict()
    return {
        "split": split_name,
        "dataset": dataset,
        "rows": len(out_df),
        "train": int(counts.get(0, 0)),
        "valid": int(counts.get(1, 0)),
        "test": int(counts.get(2, 0)),
        "target": str(target_inter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data")
    parser.add_argument("--target-root", default="DATA_SPLITS")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--splits", nargs="+", default=("random", "user_time", "global_time"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    rows = []
    for split_name in args.splits:
        for dataset in args.datasets:
            rows.append(create_split(source_root, target_root, dataset, split_name, args.seed))
    summary = pd.DataFrame(rows)
    summary_path = target_root / "three_split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("Summary written to {}".format(summary_path))


if __name__ == "__main__":
    main()
