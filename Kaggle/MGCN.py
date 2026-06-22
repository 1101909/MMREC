"""Standalone Kaggle cell for running MMRec model: MGCN.

Copy this whole file into one Kaggle notebook cell and run it.
It uses /kaggle/input/datasets/toanktx/mmrec-cold as the source data.
It prints parsed best-test metrics and writes a summary CSV after each run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


MODELS = [
    "MGCN",
]

DATASETS = ["baby", "sports", "clothing", "elec"]
DEFAULT_REPO_URL = "https://github.com/1101909/MMREC.git"
METRIC_FIELDS = [
    "Recall@5", "Recall@10", "Recall@20", "Recall@50",
    "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50",
    "Precision@5", "Precision@10", "Precision@20", "Precision@50",
    "MAP@5", "MAP@10", "MAP@20", "MAP@50",
]


RUN_ONE_CODE = r"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
import yaml

sys.path.insert(0, os.getcwd())

# Compatibility for NumPy 2.x. MMRec's metrics.py still uses np.float.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# Compatibility for newer SciPy. MMRec uses DOK.update/_update directly in a
# few models, while recent SciPy rejects direct DOK bulk updates.
def _dok_compat_update(self, data):
    if hasattr(data, "items"):
        iterator = data.items()
    else:
        iterator = data
    for key, value in iterator:
        self[key] = value
    return None


sp.dok_matrix.update = _dok_compat_update
sp.dok_matrix._update = _dok_compat_update


def normalize_config_value(value):
    if isinstance(value, list):
        return normalize_config_value(value[0]) if value else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            if any(ch in text.lower() for ch in [".", "e"]):
                return float(text)
            return int(text)
        except ValueError:
            return value
    return value


def first_values_for_model(model):
    config_path = os.path.join(os.getcwd(), "configs", "model", f"{model}.yaml")
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    hyper_parameters = set(config.get("hyper_parameters") or [])
    overrides = {}
    for key, value in config.items():
        if key in hyper_parameters:
            overrides[key] = normalize_config_value(value)
    return overrides

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="baby")
    parser.add_argument("--data-path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args, _ = parser.parse_known_args()

    from utils.quick_start import quick_start

    config_dict = {
        "gpu_id": 0,
        "use_gpu": bool(torch.cuda.is_available() and not args.cpu),
        "epochs": args.epochs,
        "eval_step": 1,
        "stopping_step": 20,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "save_recommended_topk": False,
        # Avoid running the full hyper-parameter grid for every model.
        "hyper_parameters": ["seed"],
    }
    if args.data_path:
        config_dict["data_path"] = args.data_path

    config_dict.update(first_values_for_model(args.model))
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)


if __name__ == "__main__":
    main()
"""


class KaggleMMRecPatcher:
    """Patch cloned MMRec source in-place for modern Kaggle GPU runtimes."""

    def __init__(self, src_dir: Path) -> None:
        self.src_dir = src_dir

    def apply(self) -> None:
        self._patch_utils()
        self._patch_freedom()
        self._patch_lattice()
        self._patch_mgcn()
        self._patch_smore()
        self._patch_slmrec()
        print("Source patch completed.", flush=True)

    def _read(self, rel_path: str) -> str:
        return (self.src_dir / rel_path).read_text(encoding="utf-8")

    def _write_if_changed(self, rel_path: str, text: str) -> None:
        path = self.src_dir / rel_path
        old = path.read_text(encoding="utf-8")
        if old != text:
            path.write_text(text, encoding="utf-8")

    def _patch_utils(self) -> None:
        rel_path = "utils/utils.py"
        text = self._read(rel_path)
        if "def ensure_scalar(" not in text:
            marker = "\ndef build_sim(context):\n"
            helper = r'''

def ensure_scalar(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def compute_sparse_normalized_laplacian(indices, values, shape, normalization='sym'):
    if normalization == 'none':
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    row, col = indices[0], indices[1]
    degree = torch.zeros(shape[0], dtype=values.dtype, device=values.device)
    degree.index_add_(0, row, values)

    if normalization == 'sym':
        deg_inv_sqrt = degree.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)
        values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = degree.pow(-1)
        deg_inv.masked_fill_(torch.isinf(deg_inv), 0)
        values = deg_inv[row] * values
    else:
        raise ValueError(f'Unsupported normalization: {normalization}')

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def build_knn_normalized_graph_from_embeddings(context, topk, normalization='sym', chunk_size=1024):
    device = context.device
    n_nodes = context.shape[0]
    topk = min(topk, n_nodes)
    context_norm = context / torch.clamp(torch.norm(context, p=2, dim=-1, keepdim=True), min=1e-12)
    rows, cols, vals = [], [], []

    for start in range(0, n_nodes, chunk_size):
        end = min(start + chunk_size, n_nodes)
        sim = torch.mm(context_norm[start:end], context_norm.transpose(1, 0))
        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)
        row = torch.arange(start, end, device=device).view(-1, 1).expand(-1, topk)
        rows.append(row.reshape(-1))
        cols.append(knn_ind.reshape(-1))
        vals.append(knn_val.reshape(-1))
        del sim, knn_val, knn_ind

    indices = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)
    values = torch.cat(vals)
    return compute_sparse_normalized_laplacian(indices, values, (n_nodes, n_nodes), normalization)
'''
            text = text.replace(marker, helper + marker)
        self._write_if_changed(rel_path, text)

    def _patch_freedom(self) -> None:
        rel_path = "models/freedom.py"
        text = self._read(rel_path)
        if "build_knn_normalized_graph_from_embeddings" not in text:
            text = text.replace(
                "from utils.utils import build_sim, compute_normalized_laplacian",
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_normalized_graph_from_embeddings",
            )
        old = (
            "    def get_knn_adj_mat(self, mm_embeddings):\n"
            "        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))\n"
            "        sim = torch.mm(context_norm, context_norm.transpose(1, 0))\n"
            "        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)\n"
            "        adj_size = sim.size()\n"
            "        del sim\n"
            "        # construct sparse adj\n"
            "        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)\n"
            "        indices0 = torch.unsqueeze(indices0, 1)\n"
            "        indices0 = indices0.expand(-1, self.knn_k)\n"
            "        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)\n"
            "        # norm\n"
            "        return indices, self.compute_normalized_laplacian(indices, adj_size)"
        )
        new = (
            "    def get_knn_adj_mat(self, mm_embeddings):\n"
            "        adj = build_knn_normalized_graph_from_embeddings(mm_embeddings, self.knn_k, normalization='sym')\n"
            "        adj = adj.coalesce()\n"
            "        return adj.indices(), adj"
        )
        text = text.replace(old, new)
        self._write_if_changed(rel_path, text)

    def _patch_lattice(self) -> None:
        rel_path = "models/lattice.py"
        text = self._read(rel_path)
        broken_both_modal_block = (
            "            if self.v_feat is not None and self.t_feat is not None:\n"
            "            if not learned_adj.is_sparse:\n"
            "                learned_adj = compute_normalized_laplacian(learned_adj)\n"
        )
        fixed_both_modal_block = (
            "            if self.v_feat is not None and self.t_feat is not None:\n"
            "                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj\n"
            "                original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj\n"
            "\n"
            "            if not learned_adj.is_sparse:\n"
            "                learned_adj = compute_normalized_laplacian(learned_adj)\n"
        )
        text = text.replace(broken_both_modal_block, fixed_both_modal_block)
        text = text.replace(
            "            if not learned_adj.is_sparse:\n"
            "                if not learned_adj.is_sparse:\n"
            "                    learned_adj = compute_normalized_laplacian(learned_adj)\n",
            "            if not learned_adj.is_sparse:\n"
            "                learned_adj = compute_normalized_laplacian(learned_adj)\n",
        )
        text = re.sub(
            r"(?m)^            if self\.v_feat is not None and self\.t_feat is not None:\s*\n"
            r"^            if not learned_adj\.is_sparse:\s*\n"
            r"^                learned_adj = compute_normalized_laplacian\(learned_adj\)\s*\n",
            "            if self.v_feat is not None and self.t_feat is not None:\n"
            "                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj\n"
            "                original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj\n"
            "\n"
            "            if not learned_adj.is_sparse:\n"
            "                learned_adj = compute_normalized_laplacian(learned_adj)\n",
            text,
        )
        text = re.sub(
            r"(?m)^            if self\.v_feat is not None and self\.t_feat is not None:\s*\n"
            r"^            if self\.item_adj is not None:\s*\n",
            "            if self.v_feat is not None and self.t_feat is not None:\n"
            "                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj\n"
            "                original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj\n"
            "\n"
            "            if self.item_adj is not None:\n",
            text,
        )
        if "build_knn_normalized_graph_from_embeddings" not in text:
            text = text.replace(
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood",
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph_from_embeddings",
            )
        text = text.replace(
            "image_adj_file = os.path.join(dataset_path, 'image_adj_{}.pt'.format(self.knn_k))",
            "image_adj_file = os.path.join(dataset_path, 'image_adj_sparse_{}.pt'.format(self.knn_k))",
        )
        text = text.replace(
            "text_adj_file = os.path.join(dataset_path, 'text_adj_{}.pt'.format(self.knn_k))",
            "text_adj_file = os.path.join(dataset_path, 'text_adj_sparse_{}.pt'.format(self.knn_k))",
        )
        text = text.replace(
            "image_adj = build_sim(self.image_embedding.weight.detach())\n"
            "                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)\n"
            "                image_adj = compute_normalized_laplacian(image_adj)",
            "image_adj = build_knn_normalized_graph_from_embeddings(self.image_embedding.weight.detach(), self.knn_k, normalization='sym')",
        )
        text = text.replace("self.image_original_adj = image_adj.cuda()", "self.image_original_adj = image_adj.to(self.device)")
        text = text.replace(
            "text_adj = build_sim(self.text_embedding.weight.detach())\n"
            "                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)\n"
            "                text_adj = compute_normalized_laplacian(text_adj)",
            "text_adj = build_knn_normalized_graph_from_embeddings(self.text_embedding.weight.detach(), self.knn_k, normalization='sym')",
        )
        text = text.replace("self.text_original_adj = text_adj.cuda()", "self.text_original_adj = text_adj.to(self.device)")
        text = text.replace(
            "self.image_adj = build_sim(image_feats)\n"
            "                self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.knn_k)",
            "self.image_adj = build_knn_normalized_graph_from_embeddings(image_feats, self.knn_k, normalization='sym')",
        )
        text = text.replace(
            "self.text_adj = build_sim(text_feats)\n"
            "                self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.knn_k)",
            "self.text_adj = build_knn_normalized_graph_from_embeddings(text_feats, self.knn_k, normalization='sym')",
        )
        if "if not learned_adj.is_sparse:" not in text:
            text = text.replace(
                "            learned_adj = compute_normalized_laplacian(learned_adj)\n",
                "            if not learned_adj.is_sparse:\n"
                "                learned_adj = compute_normalized_laplacian(learned_adj)\n",
            )
        text = text.replace(
            "        for i in range(self.n_layers):\n"
            "            h = torch.mm(self.item_adj, h)",
            "        for i in range(self.n_layers):\n"
            "            if self.item_adj.is_sparse:\n"
            "                h = torch.sparse.mm(self.item_adj, h)\n"
            "            else:\n"
            "                h = torch.mm(self.item_adj, h)",
        )
        self._write_if_changed(rel_path, text)

    def _patch_mgcn(self) -> None:
        rel_path = "models/mgcn.py"
        text = self._read(rel_path)
        if "build_knn_normalized_graph_from_embeddings" not in text:
            text = text.replace(
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph",
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph, build_knn_normalized_graph_from_embeddings",
            )
        text = text.replace(
            "image_adj = build_sim(self.image_embedding.weight.detach())\n"
            "                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,\n"
            "                                                       norm_type='sym')",
            "image_adj = build_knn_normalized_graph_from_embeddings(self.image_embedding.weight.detach(), self.knn_k, normalization='sym')",
        )
        text = text.replace(
            "text_adj = build_sim(self.text_embedding.weight.detach())\n"
            "                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')",
            "text_adj = build_knn_normalized_graph_from_embeddings(self.text_embedding.weight.detach(), self.knn_k, normalization='sym')",
        )
        text = text.replace("self.image_original_adj = image_adj.cuda()", "self.image_original_adj = image_adj.to(self.device)")
        text = text.replace("self.text_original_adj = text_adj.cuda()", "self.text_original_adj = text_adj.to(self.device)")
        self._write_if_changed(rel_path, text)

    def _patch_smore(self) -> None:
        rel_path = "models/smore.py"
        text = self._read(rel_path)
        if "build_knn_normalized_graph_from_embeddings" not in text:
            text = text.replace(
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph",
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph, build_knn_normalized_graph_from_embeddings, ensure_scalar",
            )
        elif "build_knn_normalized_graph_from_embeddings, ensure_scalar" not in text:
            text = text.replace(
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph, build_knn_normalized_graph_from_embeddings",
                "from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph, build_knn_normalized_graph_from_embeddings, ensure_scalar",
            )
        text = text.replace("self.reg_weight = config['reg_weight']", "self.reg_weight = ensure_scalar(config['reg_weight'])")
        text = text.replace(
            "image_adj = build_sim(self.image_embedding.weight.detach())\n"
            "                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,\n"
            "                                                       norm_type='sym')",
            "image_adj = build_knn_normalized_graph_from_embeddings(self.image_embedding.weight.detach(), self.image_knn_k, normalization='sym')",
        )
        text = text.replace(
            "text_adj = build_sim(self.text_embedding.weight.detach())\n"
            "                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')",
            "text_adj = build_knn_normalized_graph_from_embeddings(self.text_embedding.weight.detach(), self.text_knn_k, normalization='sym')",
        )
        text = text.replace("self.image_original_adj = image_adj.cuda()", "self.image_original_adj = image_adj.to(self.device)")
        text = text.replace("self.text_original_adj = text_adj.cuda()", "self.text_original_adj = text_adj.to(self.device)")
        self._write_if_changed(rel_path, text)

    def _patch_slmrec(self) -> None:
        rel_path = "models/slmrec.py"
        text = self._read(rel_path)
        if "disabled raw-text feature paths" not in text:
            text = text.replace(
                "from torch_scatter import scatter",
                "try:\n"
                "    from torch_scatter import scatter\n"
                "except ImportError:\n"
                "    def scatter(*args, **kwargs):\n"
                "        raise ImportError('torch_scatter is only required for disabled raw-text feature paths')",
            )
        self._write_if_changed(rel_path, text)


def is_mmrec_src(path: Path) -> bool:
    return (path / "configs" / "overall.yaml").exists() and (path / "main.py").exists()


def find_src_dir() -> Path | None:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd / "src",
        Path("/kaggle/working/MMRec/src"),
        Path("/kaggle/working/MMRec"),
        Path("/kaggle/working/MMREC/src"),
        Path("/kaggle/working/MMREC"),
    ]
    for candidate in candidates:
        if is_mmrec_src(candidate):
            return candidate
        src_candidate = candidate / "src"
        if is_mmrec_src(src_candidate):
            return src_candidate

    search_roots = [cwd, Path("/kaggle/working")]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for overall in root.glob("**/configs/overall.yaml"):
            src_candidate = overall.parent.parent
            if src_candidate in seen:
                continue
            seen.add(src_candidate)
            if is_mmrec_src(src_candidate):
                return src_candidate
    return None


def clone_repo(repo_url: str, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = work_dir / "MMRec"
    if not repo_dir.exists():
        print(f"Cloning {repo_url} into {repo_dir}...", flush=True)
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    src_dir = repo_dir / "src"
    if is_mmrec_src(src_dir):
        return src_dir
    if is_mmrec_src(repo_dir):
        return repo_dir
    raise FileNotFoundError(f"Cloned repo does not look like MMRec: {repo_dir}")


def ensure_python_packages(models: list[str]) -> None:
    required = {
        "yaml": "pyyaml",
        "lmdb": "lmdb",
        "PIL": "pillow",
        "torchvision": "torchvision",
    }
    if any(model in {"MMGCN", "GRCN"} for model in models):
        required["torch_geometric"] = "torch_geometric"
    missing = [pip_name for import_name, pip_name in required.items() if importlib.util.find_spec(import_name) is None]
    if missing:
        print(f"Installing missing packages: {missing}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

def is_data_root(path: Path, datasets: list[str]) -> bool:
    return path.exists() and all((path / dataset / f"{dataset}.inter").exists() for dataset in datasets)


def discover_data_path(src_dir: Path, datasets: list[str]) -> str | None:
    repo_root = src_dir.parent if src_dir.name == "src" else src_dir
    candidates = [
        Path("/kaggle/input/datasets/toanktx/mmrec-cold"),
        Path("/kaggle/working/mmrec-cold-fixed"),
        Path("/kaggle/working/DATA_COLD"),
        Path("/kaggle/working/data"),
        repo_root / "data",
        repo_root / "DATA_COLD",
        repo_root / "DATA_SPLITS" / "global_time",
        repo_root / "DATA_SPLITS" / "random",
        repo_root / "DATA_SPLITS" / "user_time",
    ]
    for candidate in candidates:
        if is_data_root(candidate, datasets):
            return str(candidate) + "/"

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.glob("*"):
            if is_data_root(candidate, datasets):
                return str(candidate) + "/"
            for child in candidate.glob("*"):
                if is_data_root(child, datasets):
                    return str(child) + "/"
    return None


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def parse_metric_line(metric_line: str | None) -> dict[str, float]:
    if not metric_line:
        return {}
    out = {}
    pairs = re.findall(r"([a-zA-Z]+@\d+)\s*:\s*([0-9]*\.?[0-9]+)", metric_line)
    for key, value in pairs:
        name, at = key.lower().split("@", 1)
        pretty = {
            "recall": "Recall",
            "ndcg": "NDCG",
            "precision": "Precision",
            "map": "MAP",
        }.get(name, name.upper())
        out[f"{pretty}@{at}"] = float(value)
    return out


def extract_result_metrics(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    valid_line = None
    test_line = None

    valid_patterns = [
        r"best valid result:\s*(recall@.*)",
        r"best valid:\s*(recall@.*)",
        r"Valid:\s*(recall@.*)",
    ]
    test_patterns = [
        r"best test:\s*(recall@.*)",
        r"test result:\s*(recall@.*)",
        r"Test:\s*(recall@.*)",
    ]

    for pattern in valid_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            valid_line = matches[-1].strip()
            break

    for pattern in test_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            test_line = matches[-1].strip()
            break

    metrics = parse_metric_line(test_line)
    row = {field: metrics.get(field, "") for field in METRIC_FIELDS}
    row["best_valid_line"] = valid_line or ""
    row["best_test_line"] = test_line or ""
    return row


def print_results_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    display_fields = ["dataset", "model", "status", "Recall@20", "NDCG@20", "Precision@20", "MAP@20"]
    print("\n===== RESULTS =====", flush=True)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        cols = [field for field in display_fields if field in df.columns]
        print(df[cols].to_string(index=False), flush=True)
    except Exception:
        print("\t".join(display_fields), flush=True)
        for row in rows:
            print("\t".join(str(row.get(field, "")) for field in display_fields), flush=True)


def validate_data_path(data_path: str | None, datasets: list[str]) -> None:
    if not data_path:
        print("WARNING: no data path detected. Use --data-path /kaggle/working/mmrec-cold-fixed/", flush=True)
        return

    root = Path(data_path)
    missing = [dataset for dataset in datasets if not (root / dataset / f"{dataset}.inter").exists()]
    if missing:
        print(f"WARNING: missing dataset folders under {root}: {missing}", flush=True)
        print("Expected structure like /kaggle/working/mmrec-cold-fixed/baby/baby.inter", flush=True)


def item_count_from_inter(inter_file: Path) -> int:
    max_item = -1
    with inter_file.open("r", encoding="utf-8") as fp:
        header = fp.readline().strip().split("\t")
        try:
            item_col = header.index("itemID")
        except ValueError:
            item_col = 1
        for line in fp:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            max_item = max(max_item, int(parts[item_col]))
    return max_item + 1


def link_or_copy(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_and_fix_dataset(source_root: Path, target_root: Path, dataset: str) -> None:
    source_dir = source_root / dataset
    target_dir = target_root / dataset
    target_dir.mkdir(parents=True, exist_ok=True)

    required = [f"{dataset}.inter"]
    optional = ["user_graph_dict.npy", "u_id_mapping.csv", "i_id_mapping.csv"]
    for name in required + optional:
        src = source_dir / name
        dst = target_dir / name
        if src.exists() and src.resolve() != dst.resolve():
            link_or_copy(src, dst)

    inter_file = target_dir / f"{dataset}.inter"
    if not inter_file.exists():
        raise FileNotFoundError(f"Missing interaction file: {inter_file}")

    n_items = item_count_from_inter(inter_file)
    for feature_name in ["image_feat.npy", "text_feat.npy"]:
        src_feature_file = source_dir / feature_name
        dst_feature_file = target_dir / feature_name
        if not src_feature_file.exists():
            continue

        feature = np.load(src_feature_file, allow_pickle=True, mmap_mode="r")
        if feature.shape[0] == n_items:
            if src_feature_file.resolve() != dst_feature_file.resolve():
                link_or_copy(src_feature_file, dst_feature_file)
            print(f"[{dataset}] {feature_name}: {feature.shape} OK", flush=True)
            continue

        old_shape = feature.shape
        if feature.shape[0] > n_items:
            feature = np.asarray(feature[:n_items])
        else:
            pad_shape = (n_items - feature.shape[0],) + feature.shape[1:]
            feature = np.concatenate([np.asarray(feature), np.zeros(pad_shape, dtype=feature.dtype)], axis=0)

        tmp_feature_file = dst_feature_file.with_name(dst_feature_file.name + ".tmp")
        with tmp_feature_file.open("wb") as fp:
            np.save(fp, feature)
        tmp_feature_file.replace(dst_feature_file)
        print(f"[{dataset}] {feature_name}: {old_shape} -> {feature.shape}", flush=True)


def prepare_data_path(data_path: str | None, datasets: list[str], work_dir: Path) -> str | None:
    if not data_path:
        return None
    source_root = Path(data_path)
    target_root = work_dir / "mmrec-cold-fixed"
    target_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        copy_and_fix_dataset(source_root, target_root, dataset)
    return str(target_root) + "/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Single dataset to run. Kept for backward compatibility.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--data-path")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--work-dir", default="/kaggle/working")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args, _ = parser.parse_known_args()

    src_dir = find_src_dir()
    if src_dir is None:
        src_dir = clone_repo(args.repo_url, Path(args.work_dir))
    print(f"MMRec src: {src_dir}", flush=True)
    ensure_python_packages(args.models)
    KaggleMMRecPatcher(src_dir).apply()
    run_dir = src_dir / "kaggle_run_results"
    run_dir.mkdir(parents=True, exist_ok=True)

    helper_path = run_dir / "_run_one_mmrec_model.py"
    helper_path.write_text(RUN_ONE_CODE, encoding="utf-8")

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = run_dir / f"summary-MGCN-{timestamp}.csv"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    datasets = [args.dataset] if args.dataset else args.datasets
    data_path = args.data_path or discover_data_path(src_dir, datasets)
    data_path = prepare_data_path(data_path, datasets, Path(args.work_dir))
    if data_path:
        print(f"Data path: {data_path}", flush=True)
    validate_data_path(data_path, datasets)

    result_rows = []
    fieldnames = [
        "dataset", "model", "status", "exit_code",
        *METRIC_FIELDS,
        "best_valid_line", "best_test_line", "log",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for dataset in datasets:
            for model in args.models:
                log_path = run_dir / f"{dataset}-{model}-{timestamp}.log"
                cmd = [
                    sys.executable,
                    str(helper_path),
                    "--model",
                    model,
                    "--dataset",
                    dataset,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--eval-batch-size",
                    str(args.eval_batch_size),
                ]
                if data_path:
                    cmd += ["--data-path", data_path]
                if args.cpu:
                    cmd += ["--cpu"]

                print(f"\n===== Running {model} on {dataset} =====", flush=True)
                with log_path.open("w", encoding="utf-8") as log_fp:
                    try:
                        proc = subprocess.run(
                            cmd,
                            cwd=str(src_dir),
                            stdout=log_fp,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                            env=env,
                        )
                        status = "PASS" if proc.returncode == 0 else "FAIL"
                        exit_code = proc.returncode
                    except subprocess.TimeoutExpired:
                        status = "TIMEOUT"
                        exit_code = 124
                        log_fp.write(f"\nTimed out after {args.timeout} seconds.\n")

                metric_row = extract_result_metrics(log_path)
                row = {
                    "dataset": dataset,
                    "model": model,
                    "status": status,
                    "exit_code": exit_code,
                    **metric_row,
                    "log": str(log_path),
                }

                writer.writerow(row)
                result_rows.append(row)
                fp.flush()
                print(f"{dataset} / {model}: {status} | log: {log_path}", flush=True)
                if row.get("best_test_line"):
                    print(
                        "metrics: "
                        f"Recall@20={row.get('Recall@20', '')} "
                        f"NDCG@20={row.get('NDCG@20', '')} "
                        f"Precision@20={row.get('Precision@20', '')} "
                        f"MAP@20={row.get('MAP@20', '')}",
                        flush=True,
                    )
                if status != "PASS":
                    print(f"----- tail {log_path.name} -----", flush=True)
                    print(tail_text(log_path), flush=True)
                    print("----- end tail -----", flush=True)

    print_results_table(result_rows)
    print(f"\nMGCN summary: {summary_path}")


if __name__ == "__main__":
    main()
