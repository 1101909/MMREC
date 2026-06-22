# coding: utf-8
"""Summarize cold-start logs for similar multimodal baselines."""

import csv
import json
import os
import re


ROOT = os.path.join("src", "run_all_results")
RUNS = {
    "baby": "20260618-160557",
    "sports": "20260618-150337",
    "clothing": "20260618-151833",
    "elec": "20260618-153720",
}
MODELS = ["BM3", "FREEDOM", "ItemKNNCBF", "MMGCN", "VBPR"]
METRICS = ["recall@20", "ndcg@20", "precision@20", "map@20", "mrr@20"]
METRIC_RE = re.compile(r"(recall@20|ndcg@20|precision@20|map@20):\s*([0-9.]+)")


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        return fp.read()


def extract_reason(text):
    markers = ("RuntimeError", "ValueError", "Traceback", "IndexError", "MemoryError")
    lines = [line.strip() for line in text.splitlines() if any(marker in line for marker in markers)]
    return lines[-1][:180] if lines else "failed/no test result"


def main():
    rows = []
    for dataset, timestamp in RUNS.items():
        for model in MODELS:
            log_path = os.path.abspath(os.path.join(ROOT, "{}-{}-cold-similar-{}.log".format(
                model, dataset, timestamp
            )))
            text = read_text(log_path)
            test_blocks = re.findall(r"test result:\s*\n?([^\n]+)", text)
            status = "PASS" if test_blocks else "FAIL"
            values = {metric: "" for metric in METRICS}
            if test_blocks:
                for key, value in METRIC_RE.findall(test_blocks[-1]):
                    values[key] = value
            rows.append({
                "dataset": dataset,
                "model": model,
                "status": status,
                **values,
                "reason": "" if status == "PASS" else extract_reason(text),
                "log": log_path,
            })

    ge_mv_path = os.path.abspath("results_baby_cpu_50ep.json")
    if os.path.exists(ge_mv_path):
        with open(ge_mv_path, "r", encoding="utf-8") as fp:
            ge_mv = json.load(fp)
        metrics_20 = ge_mv["metrics"]["20"]
        rows.append({
            "dataset": "baby",
            "model": "GE-MV-MGDP",
            "status": "PASS",
            "recall@20": "{:.4f}".format(metrics_20["recall"]),
            "ndcg@20": "{:.4f}".format(metrics_20["ndcg"]),
            "precision@20": "",
            "map@20": "",
            "mrr@20": "{:.4f}".format(metrics_20["mrr"]),
            "reason": "alpha={:.2f}, beta={:.2f}, eval_pairs={}".format(
                ge_mv["alpha"], ge_mv["beta"], ge_mv["eval_pairs"]
            ),
            "log": ge_mv_path,
        })
    for dataset in ("sports", "clothing", "elec"):
        rows.append({
            "dataset": dataset,
            "model": "GE-MV-MGDP",
            "status": "NOT_RUN",
            "recall@20": "",
            "ndcg@20": "",
            "precision@20": "",
            "map@20": "",
            "mrr@20": "",
            "reason": "not run with original threshold-graph idea on CPU; full item-item similarity is too large",
            "log": "",
        })

    out_path = os.path.abspath(os.path.join(ROOT, "all_models_with_ge_mv_cold_results_20260618.csv"))
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["dataset", "model", "status"] + METRICS + ["reason", "log"])
        writer.writeheader()
        writer.writerows(rows)

    print(out_path)
    print("dataset,model,status,recall@20,ndcg@20,precision@20,map@20,mrr@20,reason")
    for row in rows:
        print("{dataset},{model},{status},{recall@20},{ndcg@20},{precision@20},{map@20},{mrr@20},{reason}".format(**row))


if __name__ == "__main__":
    main()
