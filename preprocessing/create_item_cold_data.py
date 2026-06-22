# coding: utf-8
"""Create item-cold datasets from existing MMRec interaction files.

For each user, the latest interaction by timestamp is used as test. The
remaining interactions form a train pool. Any train-pool interaction whose item
appears in test is removed to make item-cold evaluation. Validation is then
split from the train pool by taking each user's latest remaining interaction.
Users that no longer appear in every split are removed from all splits so
train/validation/test have the same user set.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


DATASETS = ("baby", "sports", "elec", "clothing")
SIDE_FILES = (
    "image_feat.npy",
    "text_feat.npy",
    "u_id_mapping.csv",
    "i_id_mapping.csv",
    "user_graph_dict.npy",
)


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def create_cold_dataset(source_root, target_root, dataset):
    source_dir = source_root / dataset
    target_dir = target_root / dataset
    inter_name = "{}.inter".format(dataset)
    source_inter = source_dir / inter_name
    target_inter = target_dir / inter_name

    if not source_inter.is_file():
        raise FileNotFoundError(source_inter)

    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_inter, sep="\t")
    df = df.sort_values(["userID", "timestamp", "itemID"]).copy()
    df["_rank"] = df.groupby("userID").cumcount()
    df["_count"] = df.groupby("userID")["itemID"].transform("size")

    test_df = df.loc[df["_rank"] == df["_count"] - 1].copy()
    train_pool = df.loc[df["_rank"] < df["_count"] - 1].copy()

    test_items = set(test_df["itemID"].unique())
    removed_mask = train_pool["itemID"].isin(test_items)
    train_pool = train_pool.loc[~removed_mask].copy()

    train_pool = train_pool.sort_values(["userID", "timestamp", "itemID"]).copy()
    train_pool["_rank_after_cold"] = train_pool.groupby("userID").cumcount()
    train_pool["_count_after_cold"] = train_pool.groupby("userID")["itemID"].transform("size")

    valid_df = train_pool.loc[
        train_pool["_rank_after_cold"] == train_pool["_count_after_cold"] - 1
    ].copy()
    train_df = train_pool.loc[
        train_pool["_rank_after_cold"] < train_pool["_count_after_cold"] - 1
    ].copy()

    train_users = set(train_df["userID"].unique())
    valid_users = set(valid_df["userID"].unique())
    test_users = set(test_df["userID"].unique())
    kept_users = train_users & valid_users & test_users

    train_df = train_df.loc[train_df["userID"].isin(kept_users)].copy()
    valid_df = valid_df.loc[valid_df["userID"].isin(kept_users)].copy()
    test_df = test_df.loc[test_df["userID"].isin(kept_users)].copy()

    train_df["x_label"] = 0
    valid_df["x_label"] = 1
    test_df["x_label"] = 2
    output_columns = [name for name in df.columns if not name.startswith("_")]
    cold_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    cold_df = cold_df.sort_values(["userID", "x_label", "timestamp", "itemID"])
    cold_df = cold_df[output_columns]
    cold_df.to_csv(target_inter, sep="\t", index=False)

    for name in SIDE_FILES:
        src = source_dir / name
        if src.is_file():
            link_or_copy(src, target_dir / name)

    return {
        "dataset": dataset,
        "rows_before": len(df),
        "rows_after": len(cold_df),
        "train_pool_before": len(train_pool) + int(removed_mask.sum()),
        "train_after": int((cold_df["x_label"] == 0).sum()),
        "valid_after": int((cold_df["x_label"] == 1).sum()),
        "test_after": int((cold_df["x_label"] == 2).sum()),
        "removed_train_rows": int(removed_mask.sum()),
        "users_after": len(kept_users),
        "test_items": len(test_items),
        "target": str(target_inter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data")
    parser.add_argument("--target-root", default="DATA_COLD")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for dataset in args.datasets:
        summary_rows.append(create_cold_dataset(source_root, target_root, dataset))

    summary = pd.DataFrame(summary_rows)
    summary_path = target_root / "item_cold_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("Summary written to {}".format(summary_path))


if __name__ == "__main__":
    main()
